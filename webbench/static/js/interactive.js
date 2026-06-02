// Interactive single-utterance view: live per-stage bars + audio + WER.
let stageChart = null;

async function initInteractive() {
  const meta = await getJSON('/api/voices');
  const voiceSel = document.getElementById('voice');
  meta.voices.forEach(v => voiceSel.add(new Option(v, v)));
  voiceSel.value = meta.default;
  const stepSel = document.getElementById('steps');
  meta.step_choices.forEach(s => stepSel.add(new Option(`${s} steps`, s)));
  stepSel.value = meta.default_steps;

  stageChart = makeStageChart('stageChart');
  document.getElementById('generate').addEventListener('click', synth);
}

function setStage(name, ms) {
  const i = STAGE_LABELS.indexOf(name);
  if (i >= 0) { stageChart.data.datasets[0].data[i] = ms; stageChart.update(); }
}

async function synth() {
  const btn = document.getElementById('generate');
  const status = document.getElementById('status');
  const summary = document.getElementById('summary');
  const werBox = document.getElementById('werBox');
  btn.disabled = true; summary.innerHTML = ''; werBox.innerHTML = '';
  stageChart.data.datasets[0].data = new Array(7).fill(0); stageChart.update();

  const randomTake = document.getElementById('random_take').checked;
  const body = {
    text: document.getElementById('text').value,
    voice: document.getElementById('voice').value,
    steps: parseInt(document.getElementById('steps').value),
    speed: parseFloat(document.getElementById('speed').value),
    lang: 'en',
    run_wer: document.getElementById('run_wer').checked,
    // fixed seed => reproducible audio + WER; "new take" => random each run
    seed: randomTake ? null : parseInt(document.getElementById('seed').value) || 0,
  };
  status.textContent = 'Synthesizing…';

  try {
    await ssePost('/api/synth', body, {
      stage: d => {
        if (d.type === 've_step') {
          status.textContent = `Denoising ${d.step + 1}/${d.total}…`;
          // accumulate into the vector_estimator bar live
          const i = STAGE_LABELS.indexOf('vector_estimator');
          stageChart.data.datasets[0].data[i] += d.ms; stageChart.update();
        } else {
          setStage(d.type, d.ms);
        }
      },
      done: d => {
        // ensure final stage values are exact
        STAGE_LABELS.forEach(n => { if (d.stages[n] !== undefined) setStage(n, d.stages[n]); });
        document.getElementById('player').src = d.audio_url;
        const silent = d.peak < 0.01 ? ' <b style="color:#e57373">⚠ near-silent</b>' : '';
        summary.innerHTML = `Total <b>${fmt(d.total_ms)} ms</b> · audio ${fmt(d.audio_duration_s, 2)} s · RTF <b>${fmt(d.rtf, 3)}×</b>${silent}`;
        status.textContent = 'Done';
      },
      wer: d => {
        if (d.error) { werBox.innerHTML = `<span style="color:#e57373">WER error: ${d.error}</span>`; return; }
        const modeLbl = d.mode === 'absolute' ? 'absolute (vs reference)' : 'vs input text (informational)';
        werBox.innerHTML =
          `<div><span class="metric">WER <b>${fmt(d.wer, 3)}</b></span>` +
          `<span class="metric">CER <b>${fmt(d.cer, 3)}</b></span>` +
          `<span class="metric" style="color:#8b98a5">ASR ${fmt(d.asr_ms)} ms · ${d.asr_model} · ${modeLbl}</span></div>` +
          `<div class="transcript">“${d.transcript}”</div>`;
      },
      error: d => { status.textContent = 'Error: ' + d.message; },
      end: () => { btn.disabled = false; },
    });
  } catch (e) {
    status.textContent = 'Error: ' + e.message; btn.disabled = false;
  }
}
