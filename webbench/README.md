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
- **Gallery** — an on-demand, regenerable set of cached samples (fixed seed). Its
  headline is the **expression-tag A/B**: each `<sigh>`/`<laugh>`/`<breath>` phrase
  is synthesized **with** and **without** the tag, side-by-side, with the duration
  delta — so you can hear and measure how much the tag actually changes the audio.

## Verbalization toggle (num2words)

"Verbalize numbers" is an opt-in toggle (default off) that does two things:

- **TTS input** (`verbalize_input`): rewrites `$5.2M`/`21st`/`(212) 555-0142` into
  spoken words via [`verbalize.py`](verbalize.py) (regex + num2words) **before** synthesis, so the
  model reads words instead of symbols. Useful because the model's *native* handling
  of some abbreviations is weak (e.g. it spoke `$5.2M` as "5.2", dropping "million").
- **WER reference** (`verbalize_ref`): canonicalizes numbers on **both** sides of the
  WER comparison using `whisper_normalizer`'s `EnglishTextNormalizer` (after expanding
  abbreviations with the verbalizer). Whisper transcribes spoken numbers back to digit
  forms ("$5.2 million"), so a word-form reference must be canonicalized to match —
  this is what makes **absolute WER on number categories meaningful**.

Measured example ("…$5.2M…$450K…", M1, 6 steps): WER 0.235 (off) → 0.188
(verbalize input + number-normalize), and the model speaks the magnitudes clearly.

## Quality methodology

ASR round-trip WER is only meaningful in the *absolute* sense on **clean prose**
(the `harvard` category) or with the **verbalize + number-normalize** toggle on
number categories. For edge cases without verbalization there is no canonical
"spoken form", so we default to **relative WER** (compare two configs on the same
text) — this cancels the systematic ASR bias. Waveform SNR is deliberately NOT used
(the flow-matching ODE produces valid-but-time-unaligned audio, so sample-aligned
metrics are meaningless). ASR latency is always reported **separately** from
synthesis latency.

## Expression tags (finding)

`<sigh>`/`<laugh>`/`<breath>` are **not** stripped by the preprocessor (they pass
through intact and tokenize as real ids) and they **do** change the audio — but only
by ~**+170–210 ms** (`<sigh>` +169, `<laugh>` +209, `<breath>` +166; measured, seed 0).
A natural sigh/laugh is 0.3–0.8 s, so the model renders them very subtly — this is
model/checkpoint behavior, not a preprocessing bug. The Gallery makes it observable.

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

`fastapi`, `uvicorn[standard]`, `sse-starlette`, `faster-whisper`, `jiwer`,
`num2words`, `whisper_normalizer` (`onnxruntime`, `numpy`, `scipy`, `soundfile`
already present) — see [`requirements.txt`](requirements.txt). Chart.js loads from
CDN. No `torch`/`librosa` (unavailable on this Python build).

## Layout

```
app.py             FastAPI routes + lifespan model-load
config.py          paths, voices, defaults, failure thresholds
models.py          ModelBundle (sessions + text processor + voice styles + audio store)
synth.py           run_one wrapper + single-flight lock + stages_dict
verbalize.py       regex + num2words semiotic verbalizer (TTS input + WER ref)
gallery.py         on-demand sample gallery builder (incl. expression with/without)
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
