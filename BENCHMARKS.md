# Supertonic TTS — Latency Benchmarking Notes

Working notes on profiling the Supertonic 3 ONNX pipeline on CPU. Documents the
benchmark scripts, the methodology decisions made along the way, and findings
(including approaches that were tried and rejected). Results live under
`benchmark_results/` (git-ignored).

## Scripts

| Script | Purpose |
|---|---|
| `step_benchmark.py` | Per-stage latency across voices × step-counts × texts, with within-voice repetitions. Emits a text report, a `results.pkl`, and plots. |
| `visualize.py` | Reads `results.pkl` and renders the plot set. Re-runnable standalone: `python visualize.py <run_dir>/results.pkl`. |
| `extended_benchmark.py` | Independent suites: voice-variance, speed-accuracy, and a thread-count parallelism sweep. |
| `latency_benchmark.py`, `download_models.py` | Minimal usage / model-fetch examples. |

Run: `python step_benchmark.py` (from `supertonic/`). Each run writes a
timestamped subfolder under `benchmark_results/` so nothing is overwritten.

## Pipeline stages measured

`Preprocess → Duration predictor → Text encoder → Noisy-latent sampling →
Vector estimator (N-step denoising loop) → Vocoder → Audio trim`

Every reported millisecond is **full inference latency** (wall-clock sum of all
stages). There is no intra-inference streaming: the vector estimator denoises the
entire latent each step and the vocoder decodes the whole latent in one call.

## Methodology decisions

- **6 voices, not 10.** `mean ± std` over 6 voices (M1–M3, F1–F3) is statistically
  close to 10 at ~40% less runtime.
- **Reps per voice (default 5).** Repeating each voice lets us separate *within-voice*
  run-to-run noise from *across-voice* variance. The report shows per-voice CV.
- **Warmup = 0.** A warmup run does **not** help here. The cold-start cost is a
  *per-call* setup spike (see below), not a session-level warm-up that carries over.
  Verified empirically: median rep-0-vs-later-reps gap was +1% (mean +9% only because
  of a few OS-noise outliers, some of which made rep 0 *faster* — impossible for a true
  cold start). So warmup is off by default.
- **Step configs.** Default `[8, 12]`. steps=8 is the medium-quality default; 12 is the
  high end.
- **Audio saved once per voice.** Output is deterministic in *shape* per voice, so we
  save only the first rep's wav (see determinism note).

## Findings

- **The vector estimator is 86–91% of total latency** in every configuration. The
  vocoder is the only other meaningful contributor (~7–10%). Any optimization effort
  belongs in the VE loop.
- **First-step spike.** VE step 0 is **1.75–2.6× slower** than the average of steps 1+,
  in every cell. This is per-inference kernel/buffer initialization inside the loop, and
  it is *not* removed by a warmup run (each inference pays it again). The ratio shrinks
  on longer text because the fixed cost amortizes over more work.
- **Step-count cost.** Going steps 8 → 12 multiplies total time by ~1.6–2.1×.
- **Duration predictor is fully deterministic.** Same voice + speed → bitwise-identical
  predicted duration (verified: all reps `unique=1`, even across step counts). The
  latent length is derived from duration, so the latent tensor *shape* is identical every
  run.
- **Audio varies run-to-run; the source is the flow-matching noise init.** The only
  stochastic line is the noisy-latent sampling (`np.random.randn`, unseeded) that seeds
  the flow-matching ODE. The VE and vocoder are deterministic given that noise (proven
  bitwise). So repeated runs of the same voice produce the same *duration* but a
  different acoustic "take". This does **not** affect timing (identical FLOPs/shapes), so
  run-to-run timing variance is pure OS/CPU noise. Add a seed before the noisy-latent step
  if reproducible audio is needed.

## Approaches tried and rejected

- **ORT IO binding for the VE loop.** Implemented a zero-copy ping-pong-buffer denoising
  loop (output of step N bound directly as input of step N+1, constants bound once,
  `current_step` updated in place). Output was bitwise-identical to the plain path.
  **Result on `CPUExecutionProvider`: no benefit.** Across all 4 cells the paired VE-total
  delta was within ±3.4% with a std 10–20× larger than the mean, and IO binding was faster
  in only 13–15 of 30 paired runs (a coin flip). Reason: IO binding's value is eliminating
  host↔device copies, which don't exist on CPU (tensors are already in host memory, and ORT
  already wraps contiguous float32 numpy zero-copy). **Would likely help on a CUDA/DirectML
  EP** — worth revisiting there, but removed from the CPU benchmark to keep it simple.

- **Parallelizing the benchmark to reduce wall-clock.** Rejected for the *latency*
  benchmark: ORT already saturates all cores per inference, so running voices concurrently
  causes CPU contention that inflates and destabilizes per-stage timings. The only
  validity-preserving speedup is reducing the matrix (fewer voices/steps/texts). Throughput
  benchmarking is a separate exercise (see `extended_benchmark.py` parallelism suite).

## Plots produced (`<run_dir>/plots/`)

1. `1_stage_breakdown.png` — stacked per-voice stage bars (error bar = total-ms std).
2. `2_vector_estimator_per_step.png` — per-step VE latency; highlights the step-0 spike.
3. `3_total_time_distributions.png` — per-voice total-time box plots (1 box = N reps).
4. `4_cross_step_compare.png` — step-count impact per voice.

## Where the real speedup lives (CPU)

1. Fewer denoising steps (8 is already 1.6× faster than 12).
2. Quantization (INT8/FP16) of the vector estimator.
3. A GPU execution provider (where IO binding would also start to matter).
4. `intra_op_num_threads` tuning — see the parallelism suite in `extended_benchmark.py`.
