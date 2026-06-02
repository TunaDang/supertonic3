// Chart.js helpers. Dark theme defaults.
Chart.defaults.color = '#8b98a5';
Chart.defaults.borderColor = '#2a333d';

const STAGE_LABELS = ['preprocess','duration_predictor','text_encoder','noisy_latent','vector_estimator','vocoder','audio_trim'];
const STAGE_COLORS = ['#9e9e9e','#e57373','#ffb74d','#fff176','#4fc3f7','#81c784','#ba68c8'];

function makeStageChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  return new Chart(ctx, {
    type: 'bar',
    data: { labels: STAGE_LABELS, datasets: [{ label: 'ms', data: new Array(7).fill(0), backgroundColor: STAGE_COLORS }] },
    options: {
      indexAxis: 'y', animation: { duration: 250 },
      scales: { x: { title: { display: true, text: 'ms' }, beginAtZero: true } },
      plugins: { legend: { display: false } },
    },
  });
}

function makePctChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  return new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [
      { label: 'p50', data: [], backgroundColor: '#4fc3f7' },
      { label: 'p95', data: [], backgroundColor: '#ffb74d' },
      { label: 'p99', data: [], backgroundColor: '#e57373' },
    ]},
    options: { scales: { y: { beginAtZero: true, title: { display: true, text: 'ms' } } } },
  });
}

function makeWerChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  return new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [
      { label: 'WER', data: [], backgroundColor: '#4fc3f7' },
      { label: 'CER', data: [], backgroundColor: '#81c784' },
    ]},
    options: { scales: { y: { beginAtZero: true, suggestedMax: 1, title: { display: true, text: 'rate' } } } },
  });
}

function makeBarChart(canvasId, label) {
  const ctx = document.getElementById(canvasId);
  return new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: label, data: [], backgroundColor: '#4fc3f7', barThickness: 48 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, title: { display: true, text: 'ms' } } },
    },
  });
}
