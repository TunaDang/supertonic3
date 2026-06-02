// A/B compare view: same text through N configs, relative WER + latency bars.
let cmpChart = null;

async function initCompare() {
  const meta = await getJSON('/api/voices');
  const wrap = document.getElementById('cmpConfigs');
  // two default config rows: 6 steps vs 12 steps, same voice
  const mkRow = (label, voice, steps) => {
    const div = document.createElement('div');
    div.className = 'cfg-row';
    div.innerHTML =
      `<input class="label" value="${label}" />
       <select class="cv"></select>
       <select class="cs"></select>
       <label class="chk"><input type="checkbox" class="cvb"> verbalize</label>`;
    wrap.appendChild(div);
    const cv = div.querySelector('.cv'); meta.voices.forEach(v => cv.add(new Option(v, v))); cv.value = voice;
    const cs = div.querySelector('.cs'); meta.step_choices.forEach(s => cs.add(new Option(`${s} steps`, s))); cs.value = steps;
  };
  mkRow('6-step', meta.default, 6);
  mkRow('12-step', meta.default, 12);

  cmpChart = makeBarChart('cmpChart', 'total ms');
  document.getElementById('runCompare').addEventListener('click', runCompare);
}

async function runCompare() {
  const btn = document.getElementById('runCompare');
  const status = document.getElementById('cmpStatus');
  const rows = Array.from(document.querySelectorAll('#cmpConfigs .cfg-row'));
  const configs = rows.map(r => ({
    label: r.querySelector('.label').value,
    voice: r.querySelector('.cv').value,
    steps: parseInt(r.querySelector('.cs').value),
    speed: 1.05,
    verbalize_input: r.querySelector('.cvb').checked,
  }));
  const body = { text: document.getElementById('cmpText').value, lang: 'en',
                 run_wer: document.getElementById('cmp_wer').checked, configs };
  btn.disabled = true; status.textContent = 'Running…';
  try {
    const res = await postJSON('/api/compare', body);
    cmpChart.data.labels = res.results.map(r => r.label);
    cmpChart.data.datasets[0].data = res.results.map(r => r.total_ms);
    cmpChart.update();

    let html = '<table><tr><th>Config</th><th>Total ms</th><th>RTF</th><th>Audio</th><th>Transcript</th></tr>';
    res.results.forEach(r => {
      html += `<tr><td>${r.label}</td><td>${fmt(r.total_ms)}</td><td>${fmt(r.rtf, 3)}</td>` +
        `<td><audio controls src="${r.audio_url}" style="width:160px"></audio></td>` +
        `<td style="font-style:italic;color:#8b98a5">${esc(r.transcript) || '–'}</td></tr>`;
    });
    html += '</table>';
    if (res.relative && res.relative.length) {
      html += '<h3 style="margin-top:16px">Relative WER (vs first config)</h3><table><tr><th>From</th><th>To</th><th>WER</th><th>CER</th></tr>';
      res.relative.forEach(x => { html += `<tr><td>${x.from}</td><td>${x.to}</td><td>${fmt(x.wer, 3)}</td><td>${fmt(x.cer, 3)}</td></tr>`; });
      html += '</table><p class="note">Relative WER = ASR(config) vs ASR(first config) on the same text. Near 0 ⇒ the configs produce equally-intelligible audio.</p>';
    }
    document.getElementById('cmpResults').innerHTML = html;
    status.textContent = 'Done';
  } catch (e) { status.textContent = 'Error: ' + e.message; }
  btn.disabled = false;
}
