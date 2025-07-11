const express = require('express');
const path = require('path');
const fetch = require('node-fetch');
const { spawn } = require('child_process');

const numInstances = 10;
const aggregatorPort = 3000;
const startPyPorts = 8100;

const pythonProcesses = [];

for (let i = 0; i < numInstances; i++) {
  const pPort = startPyPorts + i;
  const pyProc = spawn('python', ['model.py', '--port=' + pPort], { cwd: __dirname });
  pyProc.stdout.on('data', (data) => {
    console.log('PYTHON ' + pPort + ': ' + data.toString());
  });
  pyProc.stderr.on('data', (data) => {
    console.error('PYTHON ERR ' + pPort + ': ' + data.toString());
  });
  pythonProcesses.push(pyProc);
}

const app = express();
app.use(express.json());
app.use(express.static(__dirname));

app.post('/api/chat', async (req, res) => {
  const userMsg = req.body.message || '';
  try {
    const primaryPort = startPyPorts;
    const response = await fetch('http://localhost:' + primaryPort + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userMsg })
    });
    const result = await response.json();
    return res.json({ reply: result.reply });
  } catch (err) {
    return res.json({ reply: 'Error calling main python instance: ' + err.toString() });
  }
});

app.post('/api/reason', async (req, res) => {
  const userMsg = req.body.message || '';
  try {
    const reasonReplies = [];
    for (let i = 0; i < numInstances; i++) {
      const pPort = startPyPorts + i;
      const resp = await fetch('http://localhost:' + pPort + '/reason', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: userMsg })
      });
      const j = await resp.json();
      reasonReplies.push(j.reply);
    }
    let final = combineReasoning(reasonReplies);
    return res.json({ final: final, reasonReplies: reasonReplies });
  } catch (err) {
    return res.json({ final: '', reasonReplies: [], error: err.toString() });
  }
});

function combineReasoning(replies) {
  let freq = {};
  let bestCount = 0;
  let bestReply = '';
  for (const r of replies) {
    freq[r] = (freq[r] || 0) + 1;
    if (freq[r] > bestCount) {
      bestCount = freq[r];
      bestReply = r;
    }
  }
  return bestReply;
}

app.listen(aggregatorPort, () => {
  console.log('Aggregator Node server on port ' + aggregatorPort);
  console.log('Front-end available at http://localhost:' + aggregatorPort + '/index.html');
  console.log('Spawned ' + numInstances + ' python processes on ports ' + startPyPorts + '..' + (startPyPorts + numInstances - 1));
});
