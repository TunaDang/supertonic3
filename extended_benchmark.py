"""
Extended benchmark: voice variance, speed accuracy, and parallelism tests.

Three independent test suites:
  1. voice_variance  — pick 3 outlier voices from prior results, run each 5×
  2. speed_accuracy  — test speeds 0.7 / 1.0 / 1.5 / 2.0, 5 runs each
  3. parallelism     — compare sequential vs threaded sessions at varying
                       intra_op_num_threads counts

Usage (from the supertonic/ directory):
    python extended_benchmark.py [--suite all|voice_variance|speed_accuracy|parallelism]
"""

import argparse
import concurrent.futures
import datetime
import glob
import json
import os
import platform
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "py"))
from helper import (
    UnicodeProcessor,
    Style,
    load_cfgs,
    load_onnx_all,
    load_text_processor,
    load_voice_style,
    get_latent_mask,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ONNX_DIR   = "/home/tdang1/.cache/supertonic3/onnx"
VOICE_DIR  = "/home/tdang1/.cache/supertonic3/voice_styles"
PRIOR_RESULTS = "benchmark_results/2026-05-28_initial_run/step_benchmark_results.txt"
ALL_VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

# Fixed test parameters
VARIANCE_TEXT   = (
    "Why did my $6202.39 vanish?! $NVDA & AMZN crashed 15.44% today... "
    "C'est fini! #RIP_my_wallets!"
)
VARIANCE_STEPS  = 8
VARIANCE_RUNS   = 5

SPEED_VOICE     = "M1"
SPEED_TEXT      = (
    "This morning, I took a walk in the park, and the sound of the birds "
    "and the breeze was so pleasant that I stopped for a long time just to listen. "
    "The light filtering through the trees made everything feel calm and new."
)
SPEED_STEPS     = 8
SPEED_VALUES    = [0.7, 1.0, 1.5, 2.0]
SPEED_RUNS      = 5

PARALLEL_VOICE  = "M1"
PARALLEL_TEXT   = SPEED_TEXT
PARALLEL_STEPS  = 8
PARALLEL_THREAD_COUNTS = [1, 2, 4, 7, 14]  # intra_op_num_threads
PARALLEL_RUNS   = 3   # per thread-count config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tick() -> float:
    return time.perf_counter() * 1000.0


def _stats(values):
    a = np.array(values, dtype=np.float64)
    return float(a.mean()), float(a.std()), float(a.min()), float(a.max())


def _load_session(opts, providers):
    dp, te, ve, vo = load_onnx_all(ONNX_DIR, opts, providers)
    cfgs = load_cfgs(ONNX_DIR)
    tp = load_text_processor(ONNX_DIR)
    return dp, te, ve, vo, cfgs, tp


def infer(dp_ort, text_enc_ort, vector_est_ort, vocoder_ort, cfgs,
          text_processor, text, lang, style, total_step, speed=1.05):
    """Single inference pass. Returns (wav, duration_s, timing_dict)."""
    sample_rate        = cfgs["ae"]["sample_rate"]
    base_chunk_size    = cfgs["ae"]["base_chunk_size"]
    chunk_compress     = cfgs["ttl"]["chunk_compress_factor"]
    ldim               = cfgs["ttl"]["latent_dim"]
    bsz                = 1
    t                  = {}

    t0 = _tick()
    text_ids, text_mask = text_processor([text], [lang])
    t["preprocess"] = _tick() - t0

    t0 = _tick()
    dur_onnx, *_ = dp_ort.run(None, {"text_ids": text_ids, "style_dp": style.dp,
                                      "text_mask": text_mask})
    dur_onnx /= speed
    t["duration_predictor"] = _tick() - t0

    audio_s = float(dur_onnx[0])

    t0 = _tick()
    text_emb, *_ = text_enc_ort.run(None, {"text_ids": text_ids, "style_ttl": style.ttl,
                                            "text_mask": text_mask})
    t["text_encoder"] = _tick() - t0

    t0 = _tick()
    wav_len_max  = dur_onnx.max() * sample_rate
    wav_lengths  = (dur_onnx * sample_rate).astype(np.int64)
    chunk_size   = base_chunk_size * chunk_compress
    latent_len   = int((wav_len_max + chunk_size - 1) / chunk_size)
    latent_dim   = ldim * chunk_compress
    xt           = (np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
                    * get_latent_mask(wav_lengths, base_chunk_size, chunk_compress))
    t["noisy_latent"] = _tick() - t0

    total_step_np = np.array([total_step] * bsz, dtype=np.float32)
    step_times = []
    for step in range(total_step):
        current_step = np.array([step] * bsz, dtype=np.float32)
        t0 = _tick()
        xt, *_ = vector_est_ort.run(None, {
            "noisy_latent": xt, "text_emb": text_emb, "style_ttl": style.ttl,
            "text_mask": text_mask, "latent_mask": get_latent_mask(
                wav_lengths, base_chunk_size, chunk_compress),
            "current_step": current_step, "total_step": total_step_np,
        })
        step_times.append(_tick() - t0)
    t["vector_estimator_steps"] = step_times
    t["vector_estimator_total"] = sum(step_times)

    t0 = _tick()
    wav, *_ = vocoder_ort.run(None, {"latent": xt})
    t["vocoder"] = _tick() - t0

    wav = wav[:, :int(sample_rate * audio_s)]
    t["total"] = sum(t[k] for k in ["preprocess", "duration_predictor",
                                     "text_encoder", "noisy_latent",
                                     "vector_estimator_total", "vocoder"])
    t["audio_s"] = audio_s
    t["rtf"] = (t["total"] / 1000.0) / audio_s
    return wav, audio_s, t


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outlier_voices(results_path: str, n=3) -> list:
    """
    Parse per-voice totals from a prior results file and return the n voices
    with the most extreme mean latency deviation from the group mean.
    """
    text = Path(results_path).read_text(encoding="utf-8")
    # Extract all "   V:  NNNN.NN ms  (...)" lines
    pattern = re.compile(r"^\s+(F\d|M\d):\s+([\d.]+) ms", re.MULTILINE)
    voice_totals: dict[str, list] = {}
    for m in pattern.finditer(text):
        v, ms = m.group(1), float(m.group(2))
        voice_totals.setdefault(v, []).append(ms)

    means = {v: np.mean(vals) for v, vals in voice_totals.items()}
    grand_mean = np.mean(list(means.values()))
    deviations = {v: abs(means[v] - grand_mean) for v in means}
    outliers = sorted(deviations, key=deviations.__getitem__, reverse=True)[:n]
    print(f"\n  Outlier detection (from {results_path}):")
    for v in sorted(means, key=means.__getitem__):
        marker = " ◄ OUTLIER" if v in outliers else ""
        print(f"    {v}: mean {means[v]:.0f} ms  (dev {deviations[v]:.0f} ms){marker}")
    return outliers


# ---------------------------------------------------------------------------
# Suite 1: Voice variance
# ---------------------------------------------------------------------------

def run_voice_variance(run_dir: Path, warmup: int = 1):
    print("\n" + "=" * 68)
    print("  SUITE 1: Voice Variance")
    print("=" * 68)

    outlier_voices = detect_outlier_voices(PRIOR_RESULTS, n=3)
    print(f"\n  Selected outlier voices: {outlier_voices}")
    print(f"  Text  : {VARIANCE_TEXT[:60]}...")
    print(f"  Steps : {VARIANCE_STEPS}    Runs per voice: {VARIANCE_RUNS}")

    opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    dp, te, ve, vo, cfgs, tp = _load_session(opts, providers)
    sample_rate = cfgs["ae"]["sample_rate"]

    voice_styles = {v: load_voice_style([f"{VOICE_DIR}/{v}.json"]) for v in outlier_voices}

    results = {v: [] for v in outlier_voices}

    for voice in outlier_voices:
        style = voice_styles[voice]
        print(f"\n  Voice: {voice}")

        for i in range(warmup):
            print(f"    [WARMUP {i+1}]", end=" ", flush=True)
            infer(dp, te, ve, vo, cfgs, tp, VARIANCE_TEXT, "en",
                  style, VARIANCE_STEPS)
            print("done")

        for run in range(VARIANCE_RUNS):
            print(f"    [RUN {run+1}/{VARIANCE_RUNS}]", end=" ", flush=True)
            wav, audio_s, t = infer(dp, te, ve, vo, cfgs, tp, VARIANCE_TEXT,
                                    "en", style, VARIANCE_STEPS)
            results[voice].append(t)
            print(f"  total={t['total']:.0f}ms  audio={audio_s:.2f}s  RTF={t['rtf']:.3f}x")

            audio_dir = run_dir / "audio" / "voice_variance" / voice
            audio_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(audio_dir / f"run{run+1:02d}.wav"), wav.squeeze(), sample_rate)

    # Write voice variance report
    lines = ["", "SUITE 1: Voice Variance", "=" * 68, ""]
    lines.append(f"Text    : {VARIANCE_TEXT}")
    lines.append(f"Steps   : {VARIANCE_STEPS}    Runs per voice: {VARIANCE_RUNS}")
    lines.append(f"Voices  : {outlier_voices}")
    lines.append("")

    for voice, runs in results.items():
        totals = [r["total"] for r in runs]
        audios = [r["audio_s"] for r in runs]
        rtfs   = [r["rtf"] for r in runs]
        mu, sd, mn, mx = _stats(totals)
        lines.append(f"  {voice}  total ms: {mu:.1f} ± {sd:.1f}  "
                     f"[{mn:.0f} – {mx:.0f}]  RTF {np.mean(rtfs):.3f}x  "
                     f"audio ≈ {np.mean(audios):.2f}s")
        for i, r in enumerate(runs):
            lines.append(f"    run {i+1}: {r['total']:.0f} ms  "
                         f"vecest={r['vector_estimator_total']:.0f} ms  "
                         f"vocoder={r['vocoder']:.0f} ms  "
                         f"audio={r['audio_s']:.2f}s")
        lines.append("")

    return "\n".join(lines), results


# ---------------------------------------------------------------------------
# Suite 2: Speed accuracy
# ---------------------------------------------------------------------------

def run_speed_accuracy(run_dir: Path, warmup: int = 1):
    print("\n" + "=" * 68)
    print("  SUITE 2: Speed Accuracy")
    print("=" * 68)
    print(f"  Voice : {SPEED_VOICE}   Steps: {SPEED_STEPS}   Runs: {SPEED_RUNS}")
    print(f"  Speeds: {SPEED_VALUES}")
    print(f"  Text  : {SPEED_TEXT[:60]}...")

    opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    dp, te, ve, vo, cfgs, tp = _load_session(opts, providers)
    sample_rate = cfgs["ae"]["sample_rate"]
    style = load_voice_style([f"{VOICE_DIR}/{SPEED_VOICE}.json"])

    # Get baseline audio duration at speed=1.0 (first run, not timed)
    print("\n  Measuring baseline duration at speed=1.0 ...", end=" ", flush=True)
    _, baseline_s, _ = infer(dp, te, ve, vo, cfgs, tp, SPEED_TEXT, "en",
                              style, SPEED_STEPS, speed=1.0)
    print(f"{baseline_s:.3f}s")

    results = {}

    for speed in SPEED_VALUES:
        print(f"\n  Speed={speed:.1f}")
        results[speed] = []

        for i in range(warmup):
            print(f"    [WARMUP]", end=" ", flush=True)
            infer(dp, te, ve, vo, cfgs, tp, SPEED_TEXT, "en",
                  style, SPEED_STEPS, speed=speed)
            print("done")

        for run in range(SPEED_RUNS):
            print(f"    [RUN {run+1}/{SPEED_RUNS}]", end=" ", flush=True)
            wav, audio_s, t = infer(dp, te, ve, vo, cfgs, tp, SPEED_TEXT,
                                    "en", style, SPEED_STEPS, speed=speed)
            results[speed].append(t)
            expected_s = baseline_s / speed
            error_pct  = (audio_s - expected_s) / expected_s * 100
            print(f"  audio={audio_s:.3f}s  expected={expected_s:.3f}s  "
                  f"err={error_pct:+.1f}%  total={t['total']:.0f}ms")

            audio_dir = run_dir / "audio" / "speed_accuracy" / f"speed_{speed:.1f}"
            audio_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(audio_dir / f"run{run+1:02d}.wav"), wav.squeeze(), sample_rate)

    # Write speed report
    lines = ["", "SUITE 2: Speed Accuracy", "=" * 68, ""]
    lines.append(f"Voice   : {SPEED_VOICE}   Steps: {SPEED_STEPS}")
    lines.append(f"Baseline: {baseline_s:.3f}s at speed=1.0")
    lines.append("")
    lines.append(f"  {'Speed':>6}  {'Mean audio':>11}  {'Expected':>9}  "
                 f"{'Error %':>8}  {'Dur ratio':>10}  {'Mean total':>11}  {'±Std':>8}  {'RTF':>7}")
    lines.append("  " + "-" * 76)

    for speed in SPEED_VALUES:
        runs   = results[speed]
        audios = [r["audio_s"] for r in runs]
        totals = [r["total"]   for r in runs]
        mu_a   = np.mean(audios)
        mu_t, sd_t, *_ = _stats(totals)
        expected = baseline_s / speed
        err_pct  = (mu_a - expected) / expected * 100
        ratio    = mu_a / baseline_s
        rtf      = (mu_t / 1000.0) / mu_a
        lines.append(f"  {speed:>6.1f}  {mu_a:>11.3f}  {expected:>9.3f}  "
                     f"{err_pct:>+8.2f}%  {ratio:>10.3f}  {mu_t:>11.1f}  "
                     f"{sd_t:>8.1f}  {rtf:>7.3f}x")

    lines.append("")
    lines.append("  Note: 'Dur ratio' is mean_audio / baseline. For a perfectly linear")
    lines.append("  speed parameter, ratio should equal 1/speed:")
    for speed in SPEED_VALUES:
        runs     = results[speed]
        mu_a     = np.mean([r["audio_s"] for r in runs])
        actual_r = mu_a / baseline_s
        ideal_r  = 1.0 / speed
        lines.append(f"    speed={speed:.1f}  actual ratio={actual_r:.4f}  "
                     f"ideal={ideal_r:.4f}  diff={actual_r-ideal_r:+.4f}")
    lines.append("")

    return "\n".join(lines), results


# ---------------------------------------------------------------------------
# Suite 3: Parallelism
# ---------------------------------------------------------------------------

def run_parallelism(run_dir: Path, warmup: int = 1):
    print("\n" + "=" * 68)
    print("  SUITE 3: Parallelism")
    print("=" * 68)
    cpu_count = os.cpu_count()
    print(f"  CPU count: {cpu_count}")
    print(f"  Testing intra_op_num_threads: {PARALLEL_THREAD_COUNTS}")
    print(f"  Voice: {PARALLEL_VOICE}   Steps: {PARALLEL_STEPS}   Runs: {PARALLEL_RUNS}")
    print()
    print("  Note: ORT releases the GIL during inference, so Python threads")
    print("  can genuinely run concurrently. However, N sessions × T threads")
    print("  each = N×T total threads competing for the same CPU cores.")
    print("  The sweet spot is usually (N sessions) × (cpu_count/N threads each).")

    providers = ["CPUExecutionProvider"]
    style     = load_voice_style([f"{VOICE_DIR}/{PARALLEL_VOICE}.json"])

    # ---- Part A: Single session, varying intra_op_num_threads ----
    print("\n  Part A: Single session — varying intra_op_num_threads")
    print("  " + "-" * 60)

    single_results = {}

    for n_threads in PARALLEL_THREAD_COUNTS:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        dp, te, ve, vo, cfgs, tp = _load_session(opts, providers)
        print(f"\n    threads={n_threads:2d}")

        for i in range(warmup):
            print(f"      [WARMUP]", end=" ", flush=True)
            infer(dp, te, ve, vo, cfgs, tp, PARALLEL_TEXT, "en",
                  style, PARALLEL_STEPS)
            print("done")

        runs = []
        for run in range(PARALLEL_RUNS):
            print(f"      [RUN {run+1}/{PARALLEL_RUNS}]", end=" ", flush=True)
            _, audio_s, t = infer(dp, te, ve, vo, cfgs, tp, PARALLEL_TEXT,
                                  "en", style, PARALLEL_STEPS)
            runs.append(t)
            print(f"  total={t['total']:.0f}ms  RTF={t['rtf']:.3f}x")

        single_results[n_threads] = runs

    # ---- Part B: 2 concurrent sessions in threads ----
    print("\n  Part B: 2 concurrent sessions (each with threads = cpu_count // 2)")
    print("  " + "-" * 60)

    half = max(1, cpu_count // 2)
    print(f"\n    Building 2 sessions with intra_op_num_threads={half} each ...")

    # Build 2 independent session groups
    sessions = []
    for _ in range(2):
        opts2 = ort.SessionOptions()
        opts2.intra_op_num_threads = half
        dp2, te2, ve2, vo2, cfgs2, tp2 = _load_session(opts2, providers)
        sessions.append((dp2, te2, ve2, vo2, cfgs2, tp2))

    style2 = load_voice_style([f"{VOICE_DIR}/{PARALLEL_VOICE}.json"])

    # Warmup both sessions
    print(f"    Warming up ...", end=" ", flush=True)
    for sess in sessions:
        infer(*sess, PARALLEL_TEXT, "en", style2, PARALLEL_STEPS)
    print("done")

    parallel_results = []
    seq_results      = []

    for run in range(PARALLEL_RUNS):
        # -- Sequential baseline (same two sessions, back-to-back) --
        t0_seq = _tick()
        _, _, ta = infer(*sessions[0], PARALLEL_TEXT, "en", style2, PARALLEL_STEPS)
        _, _, tb = infer(*sessions[1], PARALLEL_TEXT, "en", style2, PARALLEL_STEPS)
        seq_wall = _tick() - t0_seq
        seq_results.append({"wall": seq_wall, "a": ta, "b": tb})
        print(f"    [RUN {run+1}] sequential  wall={seq_wall:.0f}ms  "
              f"(a={ta['total']:.0f}ms, b={tb['total']:.0f}ms)")

        # -- Parallel (2 threads) --
        thread_timings = [None, None]
        thread_lock    = threading.Lock()

        def _worker(idx, sess, sty):
            _, _, t_ = infer(*sess, PARALLEL_TEXT, "en", sty, PARALLEL_STEPS)
            with thread_lock:
                thread_timings[idx] = t_

        t0_par = _tick()
        threads = [
            threading.Thread(target=_worker, args=(0, sessions[0], style2)),
            threading.Thread(target=_worker, args=(1, sessions[1], style2)),
        ]
        for thr in threads:
            thr.start()
        for thr in threads:
            thr.join()
        par_wall = _tick() - t0_par
        parallel_results.append({"wall": par_wall, "a": thread_timings[0],
                                  "b": thread_timings[1]})
        speedup = seq_wall / par_wall
        print(f"    [RUN {run+1}] parallel    wall={par_wall:.0f}ms  "
              f"(a={thread_timings[0]['total']:.0f}ms, "
              f"b={thread_timings[1]['total']:.0f}ms)  "
              f"speedup={speedup:.2f}x")

    # Write parallelism report
    lines = ["", "SUITE 3: Parallelism", "=" * 68, ""]
    lines.append(f"CPU count: {cpu_count}")
    lines.append(f"Voice: {PARALLEL_VOICE}   Steps: {PARALLEL_STEPS}   Text: {PARALLEL_TEXT[:50]}...")
    lines.append("")
    lines.append("  Part A: Single session, intra_op_num_threads sweep")
    lines.append(f"  {'Threads':>8}  {'Mean total':>12}  {'±Std':>8}  {'Min':>8}  {'Max':>8}  {'RTF':>7}")
    lines.append("  " + "-" * 58)

    baseline_ms = None
    for n_threads in PARALLEL_THREAD_COUNTS:
        runs   = single_results[n_threads]
        totals = [r["total"] for r in runs]
        mu, sd, mn, mx = _stats(totals)
        rtf = np.mean([r["rtf"] for r in runs])
        if baseline_ms is None:
            baseline_ms = mu
        speedup = baseline_ms / mu
        lines.append(f"  {n_threads:>8}  {mu:>12.1f}  {sd:>8.1f}  "
                     f"{mn:>8.1f}  {mx:>8.1f}  {rtf:>7.3f}x  (vs 1-thread: {speedup:.2f}x)")

    lines.append("")
    lines.append(f"  Part B: 2 concurrent sessions (each {half} threads)  "
                 f"vs sequential")
    lines.append(f"  {'Run':>4}  {'Seq wall':>10}  {'Par wall':>10}  "
                 f"{'Speedup':>9}  {'Par latency A':>14}  {'Par latency B':>14}")
    lines.append("  " + "-" * 66)

    for i, (s, p) in enumerate(zip(seq_results, parallel_results)):
        speedup = s["wall"] / p["wall"]
        lines.append(f"  {i+1:>4}  {s['wall']:>10.0f}  {p['wall']:>10.0f}  "
                     f"{speedup:>9.2f}x  {p['a']['total']:>14.0f}  "
                     f"{p['b']['total']:>14.0f}")

    mu_seq = np.mean([r["wall"] for r in seq_results])
    mu_par = np.mean([r["wall"] for r in parallel_results])
    mu_lat_a = np.mean([r["a"]["total"] for r in parallel_results])
    mu_lat_b = np.mean([r["b"]["total"] for r in parallel_results])
    mean_speedup = mu_seq / mu_par

    lines.append("  " + "-" * 66)
    lines.append(f"  mean  {mu_seq:>10.0f}  {mu_par:>10.0f}  "
                 f"{mean_speedup:>9.2f}x  {mu_lat_a:>14.0f}  {mu_lat_b:>14.0f}")
    lines.append("")
    lines.append("  Interpretation:")
    if mean_speedup > 1.4:
        lines.append("  ✓ Parallel sessions provide meaningful throughput gain.")
    elif mean_speedup > 1.1:
        lines.append("  ~ Modest throughput gain; per-request latency will be higher.")
    else:
        lines.append("  ✗ No throughput gain — CPU fully saturated by single session.")
    per_req_overhead = (mu_lat_a - np.mean([r["a"]["total"]
                        for r in seq_results])) / np.mean(
                        [r["a"]["total"] for r in seq_results]) * 100
    lines.append(f"  Per-request latency overhead in parallel mode: "
                 f"{per_req_overhead:+.1f}%")
    lines.append("")

    return "\n".join(lines), single_results, parallel_results, seq_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all",
                        choices=["all", "voice_variance", "speed_accuracy", "parallelism"])
    parser.add_argument("--results-dir", default="benchmark_results")
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    run_ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.results_dir) / f"{run_ts}_extended"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "extended_benchmark_results.txt"

    print("=" * 68)
    print("  Supertonic TTS — Extended Benchmark")
    print("=" * 68)
    print(f"  Suite     : {args.suite}")
    print(f"  Run dir   : {run_dir}")
    print(f"  CPU count : {os.cpu_count()}")
    print(f"  ORT       : {ort.__version__}")

    report_sections = []
    report_sections.append(
        f"SUPERTONIC TTS — Extended Benchmark\n"
        f"Run: {run_ts}\n"
        f"Platform: {platform.platform()}\n"
        f"ORT: {ort.__version__}   CPU count: {os.cpu_count()}\n"
    )

    if args.suite in ("all", "voice_variance"):
        section, _ = run_voice_variance(run_dir, args.warmup)
        report_sections.append(section)

    if args.suite in ("all", "speed_accuracy"):
        section, _ = run_speed_accuracy(run_dir, args.warmup)
        report_sections.append(section)

    if args.suite in ("all", "parallelism"):
        section, *_ = run_parallelism(run_dir, args.warmup)
        report_sections.append(section)

    report = "\n\n".join(report_sections)
    report_path.write_text(report, encoding="utf-8")

    print(f"\n\n{'='*68}")
    print(f"  Done.")
    print(f"  Report : {report_path}")
    print(f"  Audio  : {run_dir / 'audio'}/")
    print(f"{'='*68}\n")
    print(report)


if __name__ == "__main__":
    main()
