const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keyup', (e) => {
  if (e.key === 'Enter') {
    sendMessage();
  }
});

function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;
  addMessageToUI(message, 'user');
  fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: message })
  })
  .then(res => res.json())
  .then(data => {
    addMessageToUI(data.reply, 'ai');
  })
  .catch(err => {
    addMessageToUI('Error: ' + err.toString(), 'ai');
  });
  userInput.value = '';
}

function reasonAboutPrompt() {
  const message = userInput.value.trim();
  if (!message) return;
  addMessageToUI('**Reasoning** on: ' + message, 'user');
  fetch('/api/reason', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: message })
  })
  .then(res => res.json())
  .then(data => {
    addMessageToUI('Reasoning Final: ' + data.reply, 'ai');
    addMessageToUI('All replies: ' + JSON.stringify(data.reasonReplies), 'ai');
  })
  .catch(err => {
    addMessageToUI('Error reasoning: ' + err.toString(), 'ai');
  });
}

const reasonBtn = document.createElement('button');
reasonBtn.textContent = 'Reason';
reasonBtn.style.width = '80px';
reasonBtn.style.border = 'none';
reasonBtn.style.background = '#0ff';
reasonBtn.style.color = '#111';
reasonBtn.style.fontWeight = 'bold';
reasonBtn.style.cursor = 'pointer';
reasonBtn.style.marginLeft = '10px';
reasonBtn.addEventListener('click', reasonAboutPrompt);

document.getElementById('input-area').appendChild(reasonBtn);

function addMessageToUI(text, sender) {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('message');
  msgDiv.classList.add(sender);
  msgDiv.textContent = text;
  messagesDiv.appendChild(msgDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
