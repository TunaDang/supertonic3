// Gallery view: build a cached sample set; show expression with/without A/B.
// (esc() is shared from api.js)
function initGallery() {
  document.getElementById('buildGallery').addEventListener('click', buildGallery);
  loadGallery();  // show existing manifest if already built
}

async function loadGallery() {
  try {
    const m = await getJSON('/api/gallery');
    if (m.built) renderGallery(m);
  } catch (e) { /* not built yet */ }
}

async function buildGallery() {
  const btn = document.getElementById('buildGallery');
  const status = document.getElementById('galleryStatus');
  const prog = document.getElementById('galleryProg');
  btn.disabled = true; status.textContent = 'Starting…'; prog.value = 0;
  try {
    const { job_id, total } = await postJSON('/api/gallery/build', {});
    prog.max = total;
    sseGet(`/api/batch/${job_id}/events`, {
      progress: d => {
        prog.value = d.done;
        const c = d.current || {};
        status.textContent = `Synthesizing ${d.done}/${d.total} (${c.section || ''} ${c.id || ''} ${c.voice || ''}) · ${d.elapsed_s}s`;
      },
      complete: async () => {
        status.textContent = 'Loading…';
        renderGallery(await getJSON('/api/gallery'));
        status.textContent = 'Done';
        btn.disabled = false;
      },
      error: d => { status.textContent = 'Error: ' + d.error; btn.disabled = false; },
    });
  } catch (e) { status.textContent = 'Error: ' + e.message; btn.disabled = false; }
}

function aud(file) {
  return `<audio controls preload="none" src="/api/gallery/audio/${encodeURIComponent(file)}" style="width:220px"></audio>`;
}

function renderGallery(m) {
  // Expression with/without pairs
  const ex = document.getElementById('exprGallery');
  if (!m.expression_pairs || !m.expression_pairs.length) {
    ex.innerHTML = '<p class="note">No expression pairs.</p>';
  } else {
    let h = '<table><tr><th>Tag</th><th>Voice</th><th>WITH tag</th><th>WITHOUT tag</th><th>Δ duration</th></tr>';
    m.expression_pairs.forEach(p => {
      const delta = p.delta_ms;
      const color = delta > 120 ? '#81c784' : (delta > 40 ? '#ffb74d' : '#e57373');
      h += `<tr><td><b>${esc(p.tag)}</b></td><td>${esc(p.voice)}</td>` +
        `<td>${aud(p.with_file)}<br><small>${fmt(p.with_duration_s,2)}s</small></td>` +
        `<td>${aud(p.without_file)}<br><small>${fmt(p.without_duration_s,2)}s</small></td>` +
        `<td style="color:${color}"><b>${delta>=0?'+':''}${fmt(delta)} ms</b></td></tr>`;
    });
    h += '</table><p class="note">Δ is how much longer the audio is WITH the tag. Small Δ (tens of ms) ⇒ the tag barely renders; a real sigh/laugh is 300–800 ms. Listen to both to judge.</p>';
    ex.innerHTML = h;
  }

  // Sample phrases
  const sg = document.getElementById('sampleGallery');
  if (!m.items || !m.items.length) { sg.innerHTML = '<p class="note">No samples.</p>'; return; }
  let h = '<table><tr><th>Category</th><th>Text</th><th>Audio</th><th>Dur</th><th>WER</th><th>Transcript</th></tr>';
  m.items.forEach(it => {
    h += `<tr><td>${esc(it.category)}</td><td>${esc(it.text)}</td><td>${aud(it.file)}</td>` +
      `<td>${fmt(it.duration_s,2)}s</td>` +
      `<td>${it.wer==null?'–':fmt(it.wer,2)}</td>` +
      `<td style="font-style:italic;color:#8b98a5">${esc(it.transcript)||'–'}</td></tr>`;
  });
  h += '</table><p class="note">WER = ASR transcript vs the input text (after lowercase + punctuation normalization). Meaningful for clean prose; for number/symbol phrases it is confounded by verbalization + ASR — use the Interactive tab\'s "Verbalize numbers" toggle for a fair number there.</p>';
  sg.innerHTML = h;
}
