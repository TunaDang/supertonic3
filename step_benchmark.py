"""
Per-step latency benchmark for the Supertonic TTS pipeline.

Runs all 10 voices × 2 texts × 4 step configs, records mean ± std
of every stage timing across voices, and saves all audio files.

Usage (from the supertonic/ directory):
    python step_benchmark.py
    python step_benchmark.py --onnx-dir /home/tdang1/.cache/supertonic3/onnx --steps 6 8 10 12
"""

import argparse
import datetime
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    chunk_text,
)


# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------

TEXTS = {
    "short_difficult": (
        "Why did my $6202.39 vanish?! <sigh> $NVDA & AMZN crashed 15.44% today... C'est fini! <laugh> #RIP_my_wallets!"
    ),
    "paragraph": (
        "This morning, I took a walk in the park, and the sound of the birds "
        "and the breeze was so pleasant that I stopped for a long time just to listen. "
        "The light filtering through the trees made everything feel calm and new."
    ),
}

VOICES = ["M1", "M2", "M3", "F1", "F2", "F3"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepTimings:
    voice: str = ""
    text_label: str = ""
    total_steps: int = 0
    audio_duration_s: float = 0.0
    preprocess_ms: float = 0.0
    duration_predictor_ms: float = 0.0
    text_encoder_ms: float = 0.0
    noisy_latent_ms: float = 0.0
    vector_estimator_per_step_ms: list = field(default_factory=list)
    vocoder_ms: float = 0.0
    audio_trim_ms: float = 0.0

    @property
    def vector_estimator_total_ms(self) -> float:
        return sum(self.vector_estimator_per_step_ms)

    @property
    def total_ms(self) -> float:
        return (
            self.preprocess_ms
            + self.duration_predictor_ms
            + self.text_encoder_ms
            + self.noisy_latent_ms
            + self.vector_estimator_total_ms
            + self.vocoder_ms
            + self.audio_trim_ms
        )


def _stats(values: list) -> tuple:
    """Return (mean, std) for a list of floats."""
    a = np.array(values, dtype=np.float64)
    return float(a.mean()), float(a.std())


# ---------------------------------------------------------------------------
# Core instrumented inference
# ---------------------------------------------------------------------------

def _tick() -> float:
    return time.perf_counter() * 1000.0


def run_instrumented(
    cfgs: dict,
    text_processor: UnicodeProcessor,
    dp_ort: ort.InferenceSession,
    text_enc_ort: ort.InferenceSession,
    vector_est_ort: ort.InferenceSession,
    vocoder_ort: ort.InferenceSession,
    text: str,
    lang: str,
    style: Style,
    total_step: int,
    voice: str,
    text_label: str,
    speed: float = 1.05,
    verbose: bool = True,
) -> tuple:
    """Returns (wav_trimmed, StepTimings)."""
    sample_rate = cfgs["ae"]["sample_rate"]
    base_chunk_size = cfgs["ae"]["base_chunk_size"]
    chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
    ldim = cfgs["ttl"]["latent_dim"]
    bsz = 1

    t = StepTimings(voice=voice, text_label=text_label, total_steps=total_step)

    # 1. Preprocess & tokenise
    t0 = _tick()
    text_ids, text_mask = text_processor([text], [lang])
    t.preprocess_ms = _tick() - t0
    if verbose:
        print(f"    [1] Preprocess:          {t.preprocess_ms:7.2f} ms  | text_ids {text_ids.shape}")

    # 2. Duration predictor
    t0 = _tick()
    dur_onnx, *_ = dp_ort.run(
        None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
    )
    dur_onnx = dur_onnx / speed
    t.duration_predictor_ms = _tick() - t0
    t.audio_duration_s = float(dur_onnx[0])
    if verbose:
        print(f"    [2] Duration predictor:  {t.duration_predictor_ms:7.2f} ms  | {t.audio_duration_s:.2f}s audio")

    # 3. Text encoder
    t0 = _tick()
    text_emb_onnx, *_ = text_enc_ort.run(
        None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask}
    )
    t.text_encoder_ms = _tick() - t0
    if verbose:
        print(f"    [3] Text encoder:        {t.text_encoder_ms:7.2f} ms  | text_emb {text_emb_onnx.shape}")

    # 4. Noisy latent sampling
    t0 = _tick()
    wav_len_max = dur_onnx.max() * sample_rate
    wav_lengths = (dur_onnx * sample_rate).astype(np.int64)
    chunk_size = base_chunk_size * chunk_compress_factor
    latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
    latent_dim = ldim * chunk_compress_factor
    xt = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
    latent_mask = get_latent_mask(wav_lengths, base_chunk_size, chunk_compress_factor)
    xt = xt * latent_mask
    t.noisy_latent_ms = _tick() - t0
    if verbose:
        print(f"    [4] Noisy latent:        {t.noisy_latent_ms:7.2f} ms  | xt {xt.shape}")

    # 5. Vector estimator denoising loop
    total_step_np = np.array([total_step] * bsz, dtype=np.float32)
    if verbose:
        print(f"    [5] Vector estimator × {total_step} steps:")
    for step in range(total_step):
        current_step = np.array([step] * bsz, dtype=np.float32)
        t0 = _tick()
        xt, *_ = vector_est_ort.run(
            None,
            {
                "noisy_latent": xt,
                "text_emb": text_emb_onnx,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": current_step,
                "total_step": total_step_np,
            },
        )
        step_ms = _tick() - t0
        t.vector_estimator_per_step_ms.append(step_ms)
        if verbose:
            print(f"         step {step:2d}/{total_step-1}: {step_ms:7.2f} ms")
    if verbose:
        print(f"         ── total: {t.vector_estimator_total_ms:7.2f} ms  "
              f"avg/step: {t.vector_estimator_total_ms/total_step:.2f} ms")

    # 6. Vocoder
    t0 = _tick()
    wav, *_ = vocoder_ort.run(None, {"latent": xt})
    t.vocoder_ms = _tick() - t0
    if verbose:
        print(f"    [6] Vocoder:             {t.vocoder_ms:7.2f} ms  | wav {wav.shape}")

    # 7. Audio trim
    t0 = _tick()
    trim_samples = int(sample_rate * t.audio_duration_s)
    wav_trimmed = wav[:, :trim_samples]
    t.audio_trim_ms = _tick() - t0

    rtf = (t.total_ms / 1000.0) / t.audio_duration_s
    if verbose:
        print(f"    [7] Audio trim:          {t.audio_trim_ms:7.2f} ms")
        print(f"        ─────────────────────────────────────────────")
        print(f"        TOTAL: {t.total_ms:7.2f} ms  |  RTF: {rtf:.3f}x  "
              f"({'faster' if rtf < 1 else 'slower'} than real-time)")

    return wav_trimmed, t


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    results: dict,   # {(steps, text_label, voice): list[StepTimings]}
    output_path: str,
    voice_dir: str,
    n_warmup: int,
    n_reps: int,
):
    lines = []

    def w(s=""):
        lines.append(s)

    w("=" * 76)
    w("  SUPERTONIC TTS  —  Voice × Step Latency Benchmark")
    w("=" * 76)
    w()
    w(f"  Platform : {platform.platform()}")
    w(f"  Python   : {sys.version.split()[0]}")
    w(f"  ORT      : {ort.__version__}")
    w(f"  CPU count: {os.cpu_count()}")
    w(f"  Provider : CPUExecutionProvider")
    w(f"  Voices   : {', '.join(VOICES)}  (n={len(VOICES)})")
    w(f"  Reps     : {n_reps} per voice  (std reported is within-voice unless noted)")
    w(f"  Warmup   : {n_warmup} run(s) per (steps × text) combo (discarded)")
    w(f"  Audio    : saved to {voice_dir}/  (first rep only — output is deterministic per voice)")
    w()

    step_counts = sorted(set(k[0] for k in results))

    def gather(steps, text_label):
        """Return (per_voice_lists, all_flat) for this (steps, text)."""
        per_voice = []
        flat = []
        for v in VOICES:
            key = (steps, text_label, v)
            if key in results and results[key]:
                per_voice.append((v, results[key]))
                flat.extend(results[key])
        return per_voice, flat

    for text_label, text in TEXTS.items():
        w("=" * 76)
        w(f"  TEXT: {text_label}")
        w(f"  \"{text[:72]}{'...' if len(text) > 72 else ''}\"")
        w("=" * 76)
        w()

        for steps in step_counts:
            per_voice, all_t = gather(steps, text_label)
            if not all_t:
                continue
            audio_s = np.mean([t.audio_duration_s for t in all_t])
            totals = [t.total_ms for t in all_t]
            mean_total, std_total = _stats(totals)
            rtf = (mean_total / 1000.0) / audio_s

            w("─" * 76)
            w(f"  Steps = {steps}  |  voices={len(per_voice)}  reps={n_reps}  "
              f"n={len(all_t)}  |  Audio ≈ {audio_s:.2f}s  |  RTF: {rtf:.3f}x")
            w("─" * 76)
            w()
            w("  Stage stats below pool all voices × reps (n = voices × reps).")
            w()

            col = [34, 12, 12, 8]
            hdr = (f"  {'Stage':<{col[0]}}  {'Mean (ms)':>{col[1]}}  "
                   f"{'Std (ms)':>{col[2]}}  {'% total':>{col[3]}}")
            w(hdr)
            w("  " + "-" * (sum(col) + 6))

            def row(label, vals):
                mu, sd = _stats(vals)
                pct = mu / mean_total * 100 if mean_total > 0 else 0
                bar = "█" * int(pct / 3)
                w(f"  {label:<{col[0]}}  {mu:>{col[1]}.2f}  {sd:>{col[2]}.2f}  "
                  f"{pct:>{col[3]}.1f}%  {bar}")

            row("1. Preprocess & tokenise",
                [t.preprocess_ms for t in all_t])
            row("2. Duration predictor",
                [t.duration_predictor_ms for t in all_t])
            row("3. Text encoder",
                [t.text_encoder_ms for t in all_t])
            row("4. Noisy latent sampling",
                [t.noisy_latent_ms for t in all_t])

            n_vec_steps = len(all_t[0].vector_estimator_per_step_ms)
            for i in range(n_vec_steps):
                row(f"5. Vector estimator step {i:2d}/{steps-1}",
                    [t.vector_estimator_per_step_ms[i] for t in all_t])

            row("   (vector estimator TOTAL)",
                [t.vector_estimator_total_ms for t in all_t])
            row("6. Vocoder",
                [t.vocoder_ms for t in all_t])
            row("7. Audio trim",
                [t.audio_trim_ms for t in all_t])

            w("  " + "-" * (sum(col) + 6))
            w(f"  {'TOTAL':<{col[0]}}  {mean_total:>{col[1]}.2f}  "
              f"{std_total:>{col[2]}.2f}  {'100.0%':>{col[3]+1}}")
            w(f"  Min: {min(totals):.2f} ms    Max: {max(totals):.2f} ms")
            w()

            # --- Per-voice within-voice mean ± std (over reps) ---
            w(f"  Per-voice totals  (mean ± std over {n_reps} reps; CV = sd/mean):")
            w(f"    {'Voice':<6} {'Mean ms':>10} {'±Std':>8} {'CV%':>6}  "
              f"{'Audio s':>8} {'RTF':>7}  {'Min':>9} {'Max':>9}")
            for v, vt in per_voice:
                v_totals = [t.total_ms for t in vt]
                mu, sd = _stats(v_totals)
                audio_v = np.mean([t.audio_duration_s for t in vt])
                rtf_v = (mu / 1000.0) / audio_v if audio_v > 0 else 0
                cv = (sd / mu * 100) if mu > 0 else 0
                w(f"    {v:<6} {mu:>10.2f} {sd:>8.2f} {cv:>6.1f}  "
                  f"{audio_v:>8.2f} {rtf_v:>7.3f}  {min(v_totals):>9.2f} {max(v_totals):>9.2f}")
            w()

            # --- Per-voice per-stage mean ± std ---
            w(f"  Per-voice stage means (mean ± std over {n_reps} reps, ms):")
            stage_specs = [
                ("Preproc",   lambda t: t.preprocess_ms),
                ("DurPred",   lambda t: t.duration_predictor_ms),
                ("TextEnc",   lambda t: t.text_encoder_ms),
                ("NoiseLat",  lambda t: t.noisy_latent_ms),
                ("VecEst",    lambda t: t.vector_estimator_total_ms),
                ("Vocoder",   lambda t: t.vocoder_ms),
                ("Trim",      lambda t: t.audio_trim_ms),
                ("TOTAL",     lambda t: t.total_ms),
            ]
            header = "    " + f"{'Voice':<6}" + "".join(f"  {n:>16}" for n, _ in stage_specs)
            w(header)
            for v, vt in per_voice:
                cells = []
                for _, getter in stage_specs:
                    vals = [getter(t) for t in vt]
                    mu, sd = _stats(vals)
                    cells.append(f"  {mu:>7.2f}±{sd:>6.2f}")
                w(f"    {v:<6}" + "".join(cells))
            w()

        # --- Cross-step comparison for this text ---
        w("─" * 76)
        w(f"  CROSS-STEP COMPARISON  [{text_label}]")
        w("─" * 76)
        w()
        hdr2 = (f"  {'Steps':>6}  {'Mean total':>12}  {'±Std':>8}  "
                f"{'VecEst mean':>12}  {'VecEst ±Std':>12}  {'RTF':>7}  {'vs base':>8}")
        w(hdr2)
        w("  " + "-" * (len(hdr2) - 2))

        baseline_ms = None
        for steps in step_counts:
            _, all_t = gather(steps, text_label)
            if not all_t:
                continue
            audio_s = np.mean([t.audio_duration_s for t in all_t])
            mu_tot, sd_tot = _stats([t.total_ms for t in all_t])
            mu_vec, sd_vec = _stats([t.vector_estimator_total_ms for t in all_t])
            rtf = (mu_tot / 1000.0) / audio_s
            if baseline_ms is None:
                baseline_ms = mu_tot
            ratio = mu_tot / baseline_ms
            w(f"  {steps:>6}  {mu_tot:>12.2f}  {sd_tot:>8.2f}  "
              f"{mu_vec:>12.2f}  {sd_vec:>12.2f}  {rtf:>7.3f}  {ratio:>7.2f}x")
        w()

    # --- Global summary ---
    w("=" * 76)
    w("  VECTOR ESTIMATOR  —  Avg ms/step (mean ± std across all measurements)")
    w("=" * 76)
    w()
    w(f"  {'Steps':>6}  {'Text':>20}  {'Avg ms/step':>12}  {'±Std':>8}  {'VecEst % of total':>18}")
    w("  " + "-" * 70)
    for steps in step_counts:
        for text_label in TEXTS:
            _, all_t = gather(steps, text_label)
            if not all_t:
                continue
            per_step = [t.vector_estimator_total_ms / t.total_steps for t in all_t]
            mu, sd = _stats(per_step)
            pct_mu = np.mean([t.vector_estimator_total_ms / t.total_ms * 100 for t in all_t])
            w(f"  {steps:>6}  {text_label:>20}  {mu:>12.2f}  {sd:>8.2f}  {pct_mu:>17.1f}%")
    w()
    w("=" * 76)

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Voice × step latency benchmark")
    parser.add_argument(
        "--onnx-dir",
        default="/home/tdang1/.cache/supertonic3/onnx",
    )
    parser.add_argument(
        "--voice-dir",
        default="/home/tdang1/.cache/supertonic3/voice_styles",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[8, 12],
    )
    parser.add_argument("--warmup", type=int, default=0,
                        help="Discarded runs per (steps × text); per-call shape-setup cost "
                             "is paid every inference anyway, so default 0")
    parser.add_argument(
        "--reps", type=int, default=5,
        help="Timed repetitions per voice (for within-voice std)",
    )
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--results-dir", default="benchmark_results",
                        help="Root folder; each run gets its own timestamped subfolder")
    parser.add_argument("--lang", default="en")
    return parser.parse_args()


def main():
    args = parse_args()

    # Timestamped run directory — never overwrites previous results
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.results_dir) / run_ts
    audio_out = run_dir / "audio"
    report_out = run_dir / "step_benchmark_results.txt"
    run_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(args.steps) * len(TEXTS) * (args.warmup + len(VOICES) * args.reps)
    print("=" * 70)
    print("  Supertonic TTS — Voice × Step Latency Benchmark")
    print("=" * 70)
    print(f"  ONNX dir  : {args.onnx_dir}")
    print(f"  Voice dir : {args.voice_dir}")
    print(f"  Steps     : {args.steps}")
    print(f"  Voices    : {VOICES}")
    print(f"  Texts     : {list(TEXTS.keys())}")
    print(f"  Reps      : {args.reps} per voice (within-voice std)")
    print(f"  Warmup    : {args.warmup} per (steps × text)")
    print(f"  Run dir   : {run_dir}")
    print(f"  Total runs: {total_runs}  (~{total_runs * 14 // 60} min estimated)")
    print()

    # Load models once
    print("Loading models...")
    opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    cfgs = load_cfgs(args.onnx_dir)
    dp_ort, text_enc_ort, vector_est_ort, vocoder_ort = load_onnx_all(
        args.onnx_dir, opts, providers
    )
    text_processor = load_text_processor(args.onnx_dir)
    sample_rate = cfgs["ae"]["sample_rate"]
    print("Models loaded.\n")

    # Pre-load all voice styles
    voice_styles = {}
    for voice in VOICES:
        path = os.path.join(args.voice_dir, f"{voice}.json")
        voice_styles[voice] = load_voice_style([path])
    print(f"Loaded {len(voice_styles)} voice styles.\n")

    results = {}  # {(steps, text_label, voice): list[StepTimings]}  len == args.reps

    for steps in args.steps:
        for text_label, text in TEXTS.items():
            print(f"\n{'='*70}")
            print(f"  steps={steps}  text={text_label}")
            print(f"{'='*70}")

            # Warmup runs (discarded)
            for w_i in range(args.warmup):
                print(f"\n  [WARMUP {w_i+1}/{args.warmup}] voice=M1")
                run_instrumented(
                    cfgs, text_processor,
                    dp_ort, text_enc_ort, vector_est_ort, vocoder_ort,
                    text, args.lang, voice_styles["M1"],
                    steps, "M1", text_label, args.speed,
                    verbose=True,
                )

            # Timed runs — reps per voice for within-voice std
            for voice in VOICES:
                key = (steps, text_label, voice)
                results[key] = []
                for rep in range(args.reps):
                    print(f"\n  [RUN] voice={voice}  rep={rep+1}/{args.reps}  steps={steps}  text={text_label}")
                    wav, timing = run_instrumented(
                        cfgs, text_processor,
                        dp_ort, text_enc_ort, vector_est_ort, vocoder_ort,
                        text, args.lang, voice_styles[voice],
                        steps, voice, text_label, args.speed,
                        verbose=True,
                    )
                    results[key].append(timing)

                    # Save audio only on the first rep (identical text+voice → identical input,
                    # 5× disk for the same content adds no value)
                    if rep == 0:
                        audio_dir = audio_out / f"{steps}steps" / text_label
                        audio_dir.mkdir(parents=True, exist_ok=True)
                        audio_path = audio_dir / f"{voice}.wav"
                        sf.write(str(audio_path), wav.squeeze(), sample_rate)
                        print(f"    Saved: {audio_path}")

                # Per-voice within-voice summary
                vt = results[key]
                mu, sd = np.mean([t.total_ms for t in vt]), np.std([t.total_ms for t in vt])
                cv = (sd / mu * 100) if mu > 0 else 0
                print(f"\n  [{voice}] reps={args.reps}: {mu:.2f} ± {sd:.2f} ms  (CV={cv:.1f}%)")

            # Print running summary across voices+reps for this (steps, text)
            all_t = [t for v in VOICES for t in results[(steps, text_label, v)]]
            mu_tot = np.mean([t.total_ms for t in all_t])
            sd_tot = np.std([t.total_ms for t in all_t])
            mu_vec = np.mean([t.vector_estimator_total_ms for t in all_t])
            audio_s = np.mean([t.audio_duration_s for t in all_t])
            rtf = (mu_tot / 1000.0) / audio_s
            print(f"\n  Summary  steps={steps}  text={text_label}  "
                  f"voices={len(VOICES)}  reps={args.reps}  n={len(all_t)}:")
            print(f"    Total:           {mu_tot:.2f} ± {sd_tot:.2f} ms")
            print(f"    Vector estimator:{mu_vec:.2f} ms  "
                  f"({mu_vec/mu_tot*100:.1f}% of total)")
            print(f"    RTF (mean):      {rtf:.3f}x")

    # Write report
    print(f"\n\nWriting report to {report_out}...")
    report = write_report(results, str(report_out), str(audio_out), args.warmup, args.reps)

    # Save raw results for offline analysis / re-plotting
    import pickle
    pkl_path = run_dir / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "results": results,
                "voices": VOICES,
                "texts": TEXTS,
                "step_counts": args.steps,
                "n_reps": args.reps,
                "n_warmup": args.warmup,
            },
            f,
        )

    # Generate plots
    plots_dir = run_dir / "plots"
    try:
        from visualize import generate_plots
        generate_plots(pkl_path, plots_dir)
        plot_status = f"  Plots:                 {plots_dir}/"
    except Exception as e:
        plot_status = f"  Plots:                 FAILED ({e})"

    print(f"\n{'='*70}")
    print(f"  Done. Run dir:         {run_dir}")
    print(f"  Report:                {report_out}")
    print(f"  Raw data (pickle):     {pkl_path}")
    print(plot_status)
    print(f"  Audio files:           {audio_out}/")
    print(f"{'='*70}\n")
    print(report)


if __name__ == "__main__":
    main()
