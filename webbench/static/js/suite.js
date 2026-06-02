// Suite / dashboard view: run a batch over the corpus, render aggregates.
let pctChart = null, werChart = null;

async function initSuite() {
  const meta = await getJSON('/api/voices');
  const corpus = await getJSON('/api/corpus');

  const catList = document.getElementById('catList');
  catList.innerHTML = '<b style="color:#8b98a5">Categories:</b> ';
  corpus.categories.forEach(c => {
    const checked = ['harvard', 'money', 'telephone'].includes(c.id) ? 'checked' : '';
    catList.insertAdjacentHTML('beforeend',
      `<label class="chip"><input type="checkbox" class="cat" value="${c.id}" ${checked}>${c.label} (${c.n})</label>`);
  });

  const voiceList = document.getElementById('voiceList');
  voiceList.innerHTML = '<b style="color:#8b98a5">Voices:</b> ';
  meta.voices.forEach(v => {
    const checked = ['M1', 'F1'].includes(v) ? 'checked' : '';
    voiceList.insertAdjacentHTML('beforeend',
      `<label class="chip"><input type="checkbox" class="vc" value="${v}" ${checked}>${v}</label>`);
  });

  const stepList = document.getElementById('stepList');
  stepList.innerHTML = '<b style="color:#8b98a5">Steps:</b> ';
  meta.step_choices.forEach(s => {
    const checked = s === meta.default_steps ? 'checked' : '';
    stepList.insertAdjacentHTML('beforeend',
      `<label class="chip"><input type="checkbox" class="st" value="${s}" ${checked}>${s}</label>`);
  });

  pctChart = makePctChart('pctChart');
  werChart = makeWerChart('werChart');
  document.getElementById('runSuite').addEventListener('click', runSuite);
}

const checkedVals = sel => Array.from(document.querySelectorAll(sel)).filter(x => x.checked).map(x => x.value);

async function runSuite() {
  const btn = document.getElementById('runSuite');
  const status = document.getElementById('suiteStatus');
  const prog = document.getElementById('prog');
  const body = {
    categories: checkedVals('.cat'),
    voices: checkedVals('.vc'),
    steps: checkedVals('.st').map(Number),
    reps: parseInt(document.getElementById('reps').value),
    run_wer: document.getElementById('suite_wer').checked,
    speed: 1.05, lang: 'en',
  };
  btn.disabled = true; status.textContent = 'Starting…'; prog.value = 0;
  try {
    const { job_id, total } = await postJSON('/api/batch', body);
    prog.max = total;
    status.textContent = `Running 0/${total}…`;
    sseGet(`/api/batch/${job_id}/events`, {
      progress: d => {
        prog.value = d.done;
        const cur = d.current ? ` (${d.current.case_id}/${d.current.voice}/${d.current.steps}st)` : '';
        status.textContent = `Running ${d.done}/${d.total}${cur} · ${d.elapsed_s}s`;
      },
      complete: async d => {
        status.textContent = 'Aggregating…';
        const res = await getJSON(d.result_url);
        renderSuite(res);
        status.textContent = `Done · ${res.raw_count} runs`;
        btn.disabled = false;
      },
      error: d => { status.textContent = 'Error: ' + d.error; btn.disabled = false; },
    });
  } catch (e) { status.textContent = 'Error: ' + e.message; btn.disabled = false; }
}

function renderSuite(res) {
  // per-stage percentile chart
  const stages = STAGE_LABELS.concat(['total']);
  pctChart.data.labels = stages;
  pctChart.data.datasets[0].data = stages.map(s => res.per_stage[s]?.p50 ?? 0);
  pctChart.data.datasets[1].data = stages.map(s => res.per_stage[s]?.p95 ?? 0);
  pctChart.data.datasets[2].data = stages.map(s => res.per_stage[s]?.p99 ?? 0);
  pctChart.update();

  // WER by category
  const cats = Object.keys(res.per_category);
  werChart.data.labels = cats;
  werChart.data.datasets[0].data = cats.map(c => res.per_category[c].wer_mean);
  werChart.data.datasets[1].data = cats.map(c => res.per_category[c].cer_mean);
  werChart.update();

  // failures table
  const ft = document.getElementById('failTable');
  if (!res.reading_failures.length) {
    ft.innerHTML = '<p class="note">No reading failures flagged.</p>';
  } else {
    ft.innerHTML = '<table><tr><th>Case</th><th>Voice</th><th>Steps</th><th>Reason</th><th>WER</th></tr>' +
      res.reading_failures.map(f =>
        `<tr><td>${f.case_id}</td><td>${f.voice}</td><td>${f.steps}</td><td class="warn">${f.reason}</td><td>${fmt(f.wer, 2)}</td></tr>`).join('') +
      '</table>';
  }

  // per-voice table
  const vt = document.getElementById('voiceTable');
  const voices = Object.keys(res.per_voice);
  vt.innerHTML = '<table><tr><th>Voice</th><th>Total p50 (ms)</th><th>RTF p50</th><th>n</th></tr>' +
    voices.map(v => `<tr><td>${v}</td><td>${fmt(res.per_voice[v].total_ms_p50)}</td><td>${fmt(res.per_voice[v].rtf_p50, 3)}</td><td>${res.per_voice[v].n}</td></tr>`).join('') +
    '</table>';
}
