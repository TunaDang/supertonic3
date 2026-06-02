// Shared API helpers + tab switching + a POST-capable SSE reader.

// Tab switching
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} -> ${r.status} ${await r.text()}`);
  return r.json();
}

// POST + SSE: EventSource is GET-only, so parse the event stream from fetch().
// handlers: { event_name: (data) => {} }. Reserved 'end' fires when stream closes.
async function ssePost(url, body, handlers) {
  const resp = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`${url} -> ${resp.status} ${await resp.text()}`);
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
      let ev = 'message', data = '';
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) ev = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (data && handlers[ev]) {
        try { handlers[ev](JSON.parse(data)); } catch (e) { console.error(e, data); }
      }
    }
  }
  if (handlers.end) handlers.end();
}

// GET SSE via EventSource (for batch progress)
function sseGet(url, handlers) {
  const es = new EventSource(url);
  Object.entries(handlers).forEach(([ev, fn]) => {
    if (ev === 'end') return;
    es.addEventListener(ev, e => { try { fn(JSON.parse(e.data)); } catch (err) { console.error(err); } });
  });
  es.addEventListener('complete', () => es.close());
  es.addEventListener('error', () => es.close());
  return es;
}

const fmt = (x, d = 1) => (x === null || x === undefined) ? '–' : Number(x).toFixed(d);
