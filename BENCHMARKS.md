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

- **INT8 quantization of the vector estimator.** The VE dominates latency, so it is the
  obvious quantization target. Results on `CPUExecutionProvider` (Core Ultra 7 265U: AVX2 +
  AVX-VNNI, no AVX-512):
  - *Dynamic* INT8 (MatMul-only, per-tensor): **0.47× — slower.** Only the 52 MatMuls quantize;
    the 86 Convs stay FP32, so you pay fp32↔int8 conversion overhead around the quantized ops
    without quantizing the bulk of the compute.
  - *Static* INT8 (QDQ, per-tensor, MinMax calibration — quantizes Convs too): **1.78× faster,
    but audibly worse.** Confirmed by listening. The VE is an iterative flow-matching ODE
    (N steps); per-step quantization error **compounds** across the loop into a degraded
    trajectory. (Per-tensor + MinMax is the crudest config; per-channel + percentile
    calibration might recover quality, but the fewer-steps lever below makes it unnecessary.)
  - **Methodology note — waveform metrics are invalid here.** SNR / log-spectral-distance /
    max-abs-diff between FP32 and a candidate are meaningless for this pipeline: the chaotic
    ODE produces *valid-but-time-unaligned* waveforms under any perturbation. Two FP32 runs
    with *different seeds* score SNR −3.07 dB — worse than FP32-vs-INT8 at −2.44 dB. Because
    both runs are deterministic given the seed, reps can't average the difference away. Judge
    quality by listening, or by an alignment-free metric (ASR-WER / no-reference MOS), never by
    sample-aligned signal metrics.

## Plots produced (`<run_dir>/plots/`)

1. `1_stage_breakdown.png` — stacked per-voice stage bars (error bar = total-ms std).
2. `2_vector_estimator_per_step.png` — per-step VE latency; highlights the step-0 spike.
3. `3_total_time_distributions.png` — per-voice total-time box plots (1 box = N reps).
4. `4_cross_step_compare.png` — step-count impact per voice.

## Recommended setting: 6 denoising steps

**Use `total_steps = 6`.** Fewer denoising steps is the cleanest CPU speedup lever — it is
*lossless precision* (no quantization), degrades gracefully rather than cliffing, and needs no
new dependencies. The model officially supports 5 (low) to 12 (high), default 8. Listening
tests found **6 steps perceptually indistinguishable from 12**, while cutting VE latency in
roughly half. (Quantization was the alternative and was rejected — see above.)

### Measured step-count latency (6 voices × 3 reps, CPUExecutionProvider)

`short_difficult` (~10.2 s audio):

| Steps | VE mean (ms) | End-to-end (ms) | RTF | vs 12 steps |
|------:|-------------:|----------------:|----:|------------:|
| 4  | 1863 | 2183 | 0.214 | 3.3× faster |
| 5  | 2185 | 2515 | 0.247 | 2.9× faster |
| **6**  | **2767** | **3122** | **0.306** | **2.3× faster** |
| 8  | 3902 | 4348 | 0.426 | 1.65× faster |
| 12 | 6677 | 7194 | 0.706 | 1.0× (base) |

`paragraph` (~13.1 s audio):

| Steps | VE mean (ms) | End-to-end (ms) | RTF | vs 12 steps |
|------:|-------------:|----------------:|----:|------------:|
| 4  | 2281 | 2744 | 0.210 | 3.2× faster |
| 5  | 2691 | 3131 | 0.240 | 2.8× faster |
| **6**  | **3667** | **4195** | **0.321** | **2.1× faster** |
| 8  | 5242 | 5855 | 0.449 | 1.51× faster |
| 12 | 8137 | 8837 | 0.677 | 1.0× (base) |

**6 vs 12 steps: ~2.1–2.3× end-to-end** (VE ~2.2–2.4×). **6 vs the default 8: ~1.4×.** At 6
steps RTF is ~0.31 — about 3× faster than real-time playback. VE latency is ~linear in step
count plus a fixed per-inference step-0 spike (see plot 2), which is why the ratio is slightly
below the naive 12/6 = 2×.

## Other speedup levers (CPU)

1. **Fewer denoising steps** — the primary lever (above).
2. A GPU/NPU execution provider (where INT8, FP16, and IO binding would all start to matter).
3. `intra_op_num_threads` tuning — see the parallelism suite in `extended_benchmark.py`.
4. INT8 static quantization — *rejected on CPU* (faster but audibly degraded; see above).
