// server.js
const express = require('express');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const fetch = require('node-fetch');
const cors = require('cors');

const app = express();
app.use(bodyParser.json());
app.use(cors({ origin: true }));

const PORT = process.env.PORT || 3000;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || '';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY || '';
const HF_MODEL_NAME = process.env.HF_MODEL_NAME || 'google/flan-t5-small';
const BOOKING_LINK = process.env.BOOKING_LINK || '';

const providers = [];
if(GOOGLE_API_KEY) providers.push('google');
if(OPENAI_API_KEY) providers.push('openai');
if(HUGGINGFACE_API_KEY) providers.push('huggingface');
let providerIndex = 0;
function getNextProvider(){ if(providers.length===0) return null; const p = providers[providerIndex % providers.length]; providerIndex++; return p; }

// simple in-memory vector store using small TF-IDF
let vectorStore = { docs: [], vocab: [] };
function toks(s){ return String(s||'').toLowerCase().split(/\W+/).filter(Boolean); }
function buildVocab(docs, maxV=400){ const f={}; docs.forEach(d=> toks(d.text).forEach(t=> f[t]=(f[t]||0)+1)); return Object.keys(f).sort((a,b)=>f[b]-f[a]).slice(0,maxV); }
function tfvec(text,vocab){ const t=toks(text); return vocab.map(w => t.filter(x=>x===w).length); }
function dot(a,b){ let s=0; for(let i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
function norm(a){ return Math.sqrt(dot(a,a)); }
function cosine(a,b){ if(!a||!b) return 0; return dot(a,b)/(norm(a)*norm(b)+1e-12); }

app.post('/api/index', async (req,res)=>{
  try{
    const folder = path.join(__dirname,'knowledge');
    let files = [];
    if(fs.existsSync(folder)) files = fs.readdirSync(folder).filter(f => f.endsWith('.txt')||f.endsWith('.md'));
    if(files.length===0){
      const myfile = path.join(__dirname,'my_data.txt');
      if(fs.existsSync(myfile)) files = ['my_data.txt'];
    }
    if(files.length===0) return res.status(400).json({ error: 'No knowledge files found. Add .txt/.md to ./knowledge or place my_data.txt' });
    const docs = files.map(fn => ({ id: fn, text: fs.readFileSync(path.join(__dirname,fn), 'utf-8') }));
    const vocab = buildVocab(docs, 400);
    vectorStore.vocab = vocab;
    vectorStore.docs = docs.map(d=> ({ id:d.id, text:d.text, embedding: tfvec(d.text, vocab) }) );
    return res.json({ ok:true, indexed: vectorStore.docs.length });
  }catch(e){
    console.error('index error', e);
    return res.status(500).json({ error: 'index failed' });
  }
});

function retrieveTopK(query,k=3){
  if(!vectorStore.docs || vectorStore.docs.length===0) return [];
  const qv = tfvec(query, vectorStore.vocab);
  const scored = vectorStore.docs.map(d => ({ doc:d, score: cosine(qv, d.embedding) }));
  return scored.sort((a,b)=>b.score-a.score).slice(0,k).map(x=>x.doc);
}

// Provider call helpers
async function callGoogleGenerative(prompt){
  const url = `https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate?key=${GOOGLE_API_KEY}`;
  const body = { prompt: { text: prompt }, maxOutputTokens: 512 };
  const r = await fetch(url, { method:'POST', headers:{ 'Content-Type':'application/json' }, body: JSON.stringify(body) });
  if(!r.ok) throw new Error('Google error: '+(await r.text()));
  const j = await r.json();
  if(j.candidates && j.candidates[0] && j.candidates[0].output) return j.candidates[0].output;
  if(j.output) return j.output;
  return JSON.stringify(j);
}
async function callOpenAIChat(prompt){
  const url = 'https://api.openai.com/v1/chat/completions';
  const body = { model: 'gpt-4o-mini', messages:[{role:'system',content:'You are Shyam, an empathetic Indian counseling assistant.'},{role:'user',content:prompt}], max_tokens:400 };
  const r = await fetch(url,{ method:'POST', headers:{ 'Content-Type':'application/json', 'Authorization': `Bearer ${OPENAI_API_KEY}` }, body: JSON.stringify(body) });
  if(!r.ok) throw new Error('OpenAI error: '+(await r.text()));
  const j = await r.json();
  if(j.choices && j.choices[0] && j.choices[0].message) return j.choices[0].message.content;
  return JSON.stringify(j);
}
async function callHuggingFace(prompt){
  const url = `https://api-inference.huggingface.co/models/${HF_MODEL_NAME}`;
  const r = await fetch(url, { method:'POST', headers:{ 'Content-Type':'application/json', 'Authorization': `Bearer ${HUGGINGFACE_API_KEY}` }, body: JSON.stringify({ inputs: prompt, parameters:{ max_new_tokens:256 } }) });
  if(!r.ok) throw new Error('HF error: '+(await r.text()));
  const j = await r.json();
  if(Array.isArray(j) && j[0] && j[0].generated_text) return j[0].generated_text;
  if(j.generated_text) return j.generated_text;
  return JSON.stringify(j);
}

// Rotation + failover
async function generateWithRotation(prompt, maxTries = 3){
  if(providers.length===0) throw new Error('No providers configured');
  const tried = new Set(); let lastErr = null;
  for(let attempt=0; attempt<Math.min(maxTries, providers.length); attempt++){
    const provider = getNextProvider();
    if(!provider || tried.has(provider)) continue;
    tried.add(provider);
    try{
      if(provider === 'google') return await callGoogleGenerative(prompt);
      if(provider === 'openai') return await callOpenAIChat(prompt);
      if(provider === 'huggingface') return await callHuggingFace(prompt);
    }catch(e){
      console.error('provider failed', provider, e.message || e);
      lastErr = e;
      continue;
    }
  }
  throw lastErr || new Error('All providers failed');
}

// Emergency detection
const EMER_WORDS = ['suicide','kill myself','hurt myself','overdose','i want to die','self harm'];
function checkEmergency(text){ const t = String(text||'').toLowerCase(); return EMER_WORDS.some(w => t.includes(w)); }

// /api/chat
app.post('/api/chat', async (req,res)=>{
  try{
    const message = req.body && req.body.message ? String(req.body.message) : '';
    if(!message) return res.status(400).json({ error: 'Provide message' });

    if(checkEmergency(message)){
      const safety = 'I care about your safety. I cannot provide emergency services. Please call local emergency services or helplines: Vandrevala 1860-266-2345, iCall 9152987821. Would you like me to request an urgent booking?';
      return res.json({ reply: safety, emergency: true });
    }

    const top = retrieveTopK(message, 3);
    const contextText = top.map(t => `Document (${t.id}):\n${t.text}`).join('\n\n');
    const prompt = `You are Shyam, an empathetic Indian counselling assistant. Use person-centered language, validation, and culturally sensitive examples. Do NOT provide medical prescriptions. Use context below:\n\n${contextText}\n\nUser: ${message}\n\nRespond warmly and ask permission before techniques. Keep reply concise.`;

    if(providers.length>0){
      try{
        const reply = await generateWithRotation(prompt, 3);
        const result = { reply: String(reply).trim() };
        if(/\b(book|appointment|session|slot)\b/i.test(message)) result.booking = { url: BOOKING_LINK || '' };
        return res.json(result);
      }catch(e){
        console.error('generation error', e);
      }
    }

    // fallback
    const demo = `[DEMO] I hear you said: '${message}'. Retrieved docs: ${top.map(d=>d.id).join(', ') || 'none'}.`;
    return res.json({ reply: demo });
  }catch(e){
    console.error('chat handler error', e);
    return res.status(500).json({ error: 'Server error' });
  }
});

// /api/book - store booking requests (or call webhook)
app.post('/api/book', async (req,res)=>{
  try{
    const { name, phone, preferred, notes, source } = req.body || {};
    if(!name || !phone) return res.status(400).json({ error: 'Provide name and phone' });
    const bookingsFile = path.join(__dirname, 'bookings.json');
    const all = fs.existsSync(bookingsFile) ? JSON.parse(fs.readFileSync(bookingsFile,'utf-8')) : [];
    const ref = 'BK' + Date.now();
    const entry = { ref, name, phone, preferred: preferred||'', notes: notes||'', source: source||'', created: new Date().toISOString() };
    all.push(entry);
    fs.writeFileSync(bookingsFile, JSON.stringify(all, null, 2));
    // Optionally: call webhook or email admin here
    return res.json({ ok:true, ref });
  }catch(e){
    console.error('book error', e);
    return res.status(500).json({ error: 'Booking failed' });
  }
});

app.get('/', (req,res)=> res.send('Shyam backend up. POST /api/index to index, POST /api/chat to chat, POST /api/book to book.'));

app.listen(PORT, ()=> console.log('server listening on', PORT));

