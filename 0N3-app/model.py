import sys
import time
import threading
import torch
import argparse
from fastapi import FastAPI, Request
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import uvicorn
from contextlib import asynccontextmanager

class Distiller:
    def __init__(
        self,
        teacher_name="gpt2",
        student_name="distilgpt2",
        device=None,
        lr=1e-5,
        batch_size=2,
        distill_epochs=1
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_tokenizer = GPT2Tokenizer.from_pretrained(teacher_name)
        self.teacher_model = GPT2LMHeadModel.from_pretrained(teacher_name).to(self.device)
        self.teacher_model.eval()
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.teacher_model.resize_token_embeddings(len(self.teacher_tokenizer))
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_name)
        self.student_model = AutoModelForCausalLM.from_pretrained(student_name).to(self.device)
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.student_model.resize_token_embeddings(len(self.student_tokenizer))
        self.student_optimizer = optim.AdamW(self.student_model.parameters(), lr=lr)
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.batch_size = batch_size
        self.distill_epochs = distill_epochs
        self.training_data = []
        self.lock = threading.Lock()
        self.bg_thread = None
        self.bg_running = False

    def add_training_text(self, text):
        with self.lock:
            self.training_data.append(text)

    def _distill_step(self, teacher_batch, student_batch):
        with torch.no_grad():
            t_out = self.teacher_model(**teacher_batch)
        s_out = self.student_model(**student_batch)
        t_logits = t_out.logits
        s_logits = s_out.logits
        seq_len = min(t_logits.size(1), s_logits.size(1))
        t_logits = t_logits[:, :seq_len, :]
        s_logits = s_logits[:, :seq_len, :]
        t_probs = F.log_softmax(t_logits, dim=-1)
        s_probs = F.log_softmax(s_logits, dim=-1)
        loss = self.kl_div_loss(s_probs, t_probs.exp())
        self.student_optimizer.zero_grad()
        loss.backward()
        self.student_optimizer.step()
        return loss.item()

    def background_training_loop(self):
        while self.bg_running:
            with self.lock:
                if not self.training_data:
                    time.sleep(2)
                    continue
                for _ in range(self.distill_epochs):
                    idx = 0
                    while idx < len(self.training_data):
                        batch = self.training_data[idx: idx + self.batch_size]
                        idx += self.batch_size
                        if not batch:
                            break
                        t_in = self.teacher_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                        s_in = self.student_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                        self._distill_step(t_in, s_in)
                self.training_data.clear()
            time.sleep(2)

    def start_background(self):
        if not self.bg_thread:
            self.bg_running = True
            self.bg_thread = threading.Thread(target=self.background_training_loop, daemon=True)
            self.bg_thread.start()

    def stop_background(self):
        self.bg_running = False
        if self.bg_thread:
            self.bg_thread.join()
        self.bg_thread = None

    def generate(self, prompt, max_length=60, temperature=1.0, repetition_penalty=1.2):
        with self.lock:
            shaped_prompt = f"User: {prompt}\nAI:"
            enc = self.student_tokenizer(shaped_prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                out = self.student_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.student_tokenizer.pad_token_id
                )
            txt = self.student_tokenizer.decode(out[0], skip_special_tokens=True)
            if "AI:" in txt:
                txt = txt.split("AI:", 1)[-1].strip()
            return txt

app = FastAPI()
distiller = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global distiller
    distiller = Distiller()
    distiller.start_background()
    yield
    distiller.stop_background()

app.router.lifespan_context = lifespan

@app.post("/chat")
async def chat_endpoint(request: Request):
    global distiller
    data = await request.json()
    user_text = data.get("message", "")
    if not user_text.strip():
        return {"reply": "I didn't catch that."}
    distiller.add_training_text(user_text)
    reply = distiller.generate(user_text, max_length=80, temperature=0.9, repetition_penalty=1.2)
    return {"reply": reply}

@app.post("/reason")
async def reason_endpoint(request: Request):
    global distiller
    data = await request.json()
    prompt = data.get("prompt", "")
    if not prompt.strip():
        return {"reply": "No prompt."}
    reply = distiller.generate(f"Reasoning steps about: {prompt}", max_length=100, temperature=1.0, repetition_penalty=1.1)
    return {"reply": reply}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__=="__main__":
    main()
