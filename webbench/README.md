# Supertonic TTS Benchmark — Web Tool

A Python-backed web frontend for benchmarking the Supertonic ONNX TTS pipeline.
Reuses the per-stage instrumentation in [`../step_benchmark.py`](../step_benchmark.py)
(`run_instrumented` + `StepTimings`) and adds ASR round-trip WER/CER via
`faster-whisper`.

## Features

- **Interactive** — type a sentence; watch the 7 pipeline stages fill a bar graph
  **live** over SSE as synthesis runs; play the audio; optionally compute WER/CER
  by transcribing the output back with Whisper.
- **Suite / Dashboard** — run an expanded English edge-case corpus (Harvard clean
  prose, the 16 text-normalization semiotic classes, homographs, prosody,
  expression tags, robustness/abuse) × voices × step counts and get aggregate
  latency percentiles (p50/p95/p99) per stage, WER by category, reading-failure
  flags, and per-voice medians.
- **A/B Compare** — same text through N configs (e.g. 6-step vs 12-step); shows
  latency side-by-side + **relative WER** (ASR of config A vs ASR of config B).

## Quality methodology

ASR round-trip WER is only meaningful in the *absolute* sense on **clean prose**
(the `harvard` category, where the reference is the exact text). For edge cases
(money, dates, …) there is no canonical "spoken form", so we use **relative WER**
(compare two configs on the same text) — this cancels the systematic ASR bias.
Waveform SNR is deliberately NOT used (the flow-matching ODE produces
valid-but-time-unaligned audio, so sample-aligned metrics are meaningless). ASR
latency is always reported **separately** from synthesis latency.

## Run

```bash
# from the supertonic/ repo root, using the project venv
cd webbench
PYTHONDONTWRITEBYTECODE=1 /home/tdang1/.venv_tts/bin/python -m uvicorn app:app \
    --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

- Models load once at startup (~30–60 s, ~360 MB ONNX). Do **not** use `--reload`.
- `faster-whisper base.en` (~140 MB) downloads on the **first** WER request.
- Synthesis is single-flight (`SYNTH_LOCK`) — ONNX sessions aren't safe for
  concurrent `.run()`, and serialization keeps timings clean.

## Dependencies (added to the venv)

`fastapi`, `uvicorn[standard]`, `sse-starlette`, `faster-whisper`, `jiwer`
(`onnxruntime`, `numpy`, `scipy`, `soundfile` already present). Chart.js loads
from CDN. No `torch`/`librosa` (unavailable on this Python build).

## Layout

```
app.py             FastAPI routes + lifespan model-load
config.py          paths, voices, defaults, failure thresholds
models.py          ModelBundle (sessions + text processor + voice styles + audio store)
synth.py           run_one wrapper + single-flight lock + stages_dict
pipeline_stream.py SSE bridge (threadpool + asyncio.Queue) for /api/synth
wer.py             WhisperScorer (faster-whisper + jiwer, 44.1k->16k resample)
corpus.py          English edge-case corpus
batch.py, jobs.py  background suite runner + aggregation + job registry
audio.py, schemas.py
static/            index.html + css + vanilla JS (Chart.js)
```

## Notes / caveats

- Expression-tag cases (`<sigh>`, `<laugh>`) are **observational** — the text
  preprocessor passes `< >` through but whether the model renders them is a model
  question, not a pass/fail assertion.
- Batch latency is sensitive to CPU thermal throttling under sustained load; the
  dashboard reports percentiles (not just means) for this reason.
- Whisper `base.en` is itself an imperfect oracle (ASR errors inflate absolute
  WER); relative A/B WER largely cancels that bias.
