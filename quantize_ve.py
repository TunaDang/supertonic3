"""
INT8 quantization experiment for the Supertonic vector estimator.

The vector estimator is 86-91% of CPU inference latency, so it is the single
biggest latency lever. This script quantizes it to INT8 and measures both the
latency win and the quality cost.

Quality is measured by SEEDING the flow-matching noise so the FP32 and INT8
pipelines denoise the *same* starting latent — otherwise the stochastic per-run
"take" variation would swamp the quantization delta. We then compare the two
output waveforms directly (SNR + log-spectral distance) and save both for A/B
listening.

Usage (from supertonic/):
    python quantize_ve.py                       # dynamic quant, 6 voices
    python quantize_ve.py --mode static         # static quant w/ calibration
    python quantize_ve.py --voices M1 F1 --reps 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import stft

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "py"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper import (load_cfgs, load_onnx, load_onnx_all, load_text_processor,
                    load_voice_style, get_latent_mask)
import step_benchmark as sb


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def _try_pre_process(fp32_path, pre_path):
    """Best-effort shape-inference preprocess; return the path to quantize.

    Dynamic-shape models can fail symbolic shape inference; dynamic quant does
    not require it, so fall back to the raw model.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process
    try:
        quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=True)
        return str(pre_path)
    except Exception as e:
        print(f"  (pre-process skipped: {type(e).__name__})")
        return str(fp32_path)


def make_dynamic_int8(fp32_path, int8_path, include_conv=False):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    pre = str(int8_path) + ".pre.onnx"
    src = _try_pre_process(fp32_path, pre)
    op_types = ["MatMul", "Conv"] if include_conv else None
    kwargs = dict(weight_type=QuantType.QInt8)
    if op_types:
        kwargs["op_types_to_quantize"] = op_types
    quantize_dynamic(src, str(int8_path), **kwargs)
    if os.path.exists(pre):
        try:
            os.remove(pre)
        except OSError:
            pass


class _VECalibrationReader:
    """Feeds recorded VE inputs to the static quantizer."""
    def __init__(self, samples):
        self._it = iter(samples)

    def get_next(self):
        return next(self._it, None)


def make_static_int8(fp32_path, int8_path, calib_samples):
    from onnxruntime.quantization import (quantize_static, QuantType,
                                          CalibrationMethod, QuantFormat)
    pre = str(int8_path) + ".pre.onnx"
    src = _try_pre_process(fp32_path, pre)
    quantize_static(
        src, str(int8_path),
        _VECalibrationReader(calib_samples),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    if os.path.exists(pre):
        try:
            os.remove(pre)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Quality metrics (waveform level)
# ---------------------------------------------------------------------------

def snr_db(ref, test):
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    noise = ref - test
    p_ref = float(np.mean(ref ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise == 0:
        return float("inf")
    return 10.0 * np.log10(p_ref / p_noise)


def log_spectral_distance(ref, test, sr):
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    _, _, Zr = stft(ref, fs=sr, nperseg=1024, noverlap=768)
    _, _, Zt = stft(test, fs=sr, nperseg=1024, noverlap=768)
    Sr = np.log(np.abs(Zr) ** 2 + 1e-10)
    St = np.log(np.abs(Zt) ** 2 + 1e-10)
    return float(np.sqrt(np.mean((Sr - St) ** 2)))


def max_abs_diff(ref, test):
    n = min(len(ref), len(test))
    return float(np.max(np.abs(ref[:n] - test[:n])))


# ---------------------------------------------------------------------------
# Calibration data collection (for static mode)
# ---------------------------------------------------------------------------

def collect_calibration(cfgs, tp, dp, te, voice_styles, voices, text, lang,
                        steps, speed, seed0=100):
    """Run FP32 VE once per voice and record every VE input dict it sees."""
    samples = []
    sample_rate = cfgs["ae"]["sample_rate"]
    base_chunk = cfgs["ae"]["base_chunk_size"]
    ccf = cfgs["ttl"]["chunk_compress_factor"]
    ldim = cfgs["ttl"]["latent_dim"]
    for i, v in enumerate(voices):
        np.random.seed(seed0 + i)
        style = voice_styles[v]
        text_ids, text_mask = tp([text], [lang])
        dur, *_ = dp.run(None, {"text_ids": text_ids, "style_dp": style.dp,
                                "text_mask": text_mask})
        dur = dur / speed
        text_emb, *_ = te.run(None, {"text_ids": text_ids, "style_ttl": style.ttl,
                                     "text_mask": text_mask})
        wav_len_max = dur.max() * sample_rate
        wav_lengths = (dur * sample_rate).astype(np.int64)
        chunk_size = base_chunk * ccf
        latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
        xt = np.random.randn(1, ldim * ccf, latent_len).astype(np.float32)
        latent_mask = get_latent_mask(wav_lengths, base_chunk, ccf)
        xt = xt * latent_mask
        total_step_np = np.array([steps], dtype=np.float32)
        # record one input per step (a handful is plenty for MinMax calibration)
        for step in range(steps):
            samples.append({
                "noisy_latent": xt.astype(np.float32),
                "text_emb": text_emb,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": np.array([step], dtype=np.float32),
                "total_step": total_step_np,
            })
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="INT8 quantize the vector estimator")
    p.add_argument("--onnx-dir", default="/home/tdang1/.cache/supertonic3/onnx")
    p.add_argument("--voice-dir", default="/home/tdang1/.cache/supertonic3/voice_styles")
    p.add_argument("--voices", nargs="+", default=["M1", "M2", "M3", "F1", "F2", "F3"])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--reps", type=int, default=3, help="latency reps (quality is seeded/deterministic)")
    p.add_argument("--speed", type=float, default=1.05)
    p.add_argument("--lang", default="en")
    p.add_argument("--text", default=sb.TEXTS["short_difficult"])
    p.add_argument("--mode", choices=["dynamic", "dynamic_conv", "static"], default="dynamic")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", default="quant_results")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / args.mode
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    fp32_ve_path = os.path.join(args.onnx_dir, "vector_estimator.onnx")
    int8_ve_path = out_dir / "vector_estimator_int8.onnx"

    cfgs = load_cfgs(args.onnx_dir)
    sample_rate = cfgs["ae"]["sample_rate"]
    opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]

    dp, te, ve_fp32, vo = load_onnx_all(args.onnx_dir, opts, providers)
    tp = load_text_processor(args.onnx_dir)
    voice_styles = {v: load_voice_style([os.path.join(args.voice_dir, f"{v}.json")])
                    for v in args.voices}

    print(f"Quantizing VE  ({args.mode})  ...")
    t0 = time.perf_counter()
    if args.mode == "dynamic":
        make_dynamic_int8(fp32_ve_path, int8_ve_path, include_conv=False)
    elif args.mode == "dynamic_conv":
        make_dynamic_int8(fp32_ve_path, int8_ve_path, include_conv=True)
    else:
        calib = collect_calibration(cfgs, tp, dp, te, voice_styles, args.voices,
                                    args.text, args.lang, args.steps, args.speed)
        print(f"  collected {len(calib)} calibration samples")
        make_static_int8(fp32_ve_path, int8_ve_path, calib)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    fp_mb = os.path.getsize(fp32_ve_path) / 1e6
    q_mb = os.path.getsize(int8_ve_path) / 1e6
    print(f"  VE size: {fp_mb:.1f} MB -> {q_mb:.1f} MB  ({q_mb/fp_mb*100:.0f}%)\n")

    ve_int8 = load_onnx(str(int8_ve_path), opts, providers)

    # ---- compare ----
    rows = []
    for i, v in enumerate(args.voices):
        style = voice_styles[v]
        seed = args.seed + i  # distinct per voice, identical across the two modes

        # FP32: quality run (seeded) + latency reps
        np.random.seed(seed)
        wav_fp, t_fp = sb.run_instrumented(
            cfgs, tp, dp, te, ve_fp32, vo, args.text, args.lang, style,
            args.steps, v, "quant", args.speed, verbose=False)
        ve_fp_reps = [t_fp.vector_estimator_total_ms]
        for _ in range(args.reps - 1):
            np.random.seed(seed)
            _, tt = sb.run_instrumented(cfgs, tp, dp, te, ve_fp32, vo, args.text,
                                        args.lang, style, args.steps, v, "quant",
                                        args.speed, verbose=False)
            ve_fp_reps.append(tt.vector_estimator_total_ms)

        # INT8: quality run (same seed) + latency reps
        np.random.seed(seed)
        wav_q, t_q = sb.run_instrumented(
            cfgs, tp, dp, te, ve_int8, vo, args.text, args.lang, style,
            args.steps, v, "quant", args.speed, verbose=False)
        ve_q_reps = [t_q.vector_estimator_total_ms]
        for _ in range(args.reps - 1):
            np.random.seed(seed)
            _, tt = sb.run_instrumented(cfgs, tp, dp, te, ve_int8, vo, args.text,
                                        args.lang, style, args.steps, v, "quant",
                                        args.speed, verbose=False)
            ve_q_reps.append(tt.vector_estimator_total_ms)

        a = wav_fp.squeeze().astype(np.float64)
        b = wav_q.squeeze().astype(np.float64)
        snr = snr_db(a, b)
        lsd = log_spectral_distance(a, b, sample_rate)
        mad = max_abs_diff(a, b)

        ve_fp = float(np.mean(ve_fp_reps))
        ve_q = float(np.mean(ve_q_reps))
        speedup = ve_fp / ve_q if ve_q > 0 else 0.0

        sf.write(str(audio_dir / f"{v}_fp32.wav"), wav_fp.squeeze(), sample_rate)
        sf.write(str(audio_dir / f"{v}_int8.wav"), wav_q.squeeze(), sample_rate)

        rows.append((v, ve_fp, ve_q, speedup, snr, lsd, mad))
        print(f"  {v}:  VE {ve_fp:7.1f} -> {ve_q:7.1f} ms  ({speedup:.2f}x)  "
              f"SNR {snr:6.2f} dB  LSD {lsd:.3f}  maxΔ {mad:.4f}")

    # ---- summary ----
    ve_fp_all = np.array([r[1] for r in rows])
    ve_q_all = np.array([r[2] for r in rows])
    snrs = np.array([r[4] for r in rows])
    lsds = np.array([r[5] for r in rows])
    overall_speedup = ve_fp_all.mean() / ve_q_all.mean()

    lines = []
    lines.append("=" * 76)
    lines.append(f"  VECTOR ESTIMATOR INT8 QUANTIZATION  —  mode={args.mode}")
    lines.append("=" * 76)
    lines.append(f"  Platform : {__import__('platform').platform()}")
    lines.append(f"  ORT      : {ort.__version__}   CPU: {os.cpu_count()}")
    lines.append(f"  Voices   : {', '.join(args.voices)}   steps={args.steps}   reps={args.reps}")
    lines.append(f"  VE size  : {fp_mb:.1f} MB -> {q_mb:.1f} MB  ({q_mb/fp_mb*100:.0f}%)")
    lines.append("")
    lines.append(f"  {'Voice':<6} {'VE fp32':>9} {'VE int8':>9} {'speedup':>8} "
                 f"{'SNR dB':>8} {'LSD':>7} {'maxΔ':>8}")
    lines.append("  " + "-" * 60)
    for v, ve_fp, ve_q, sp, snr, lsd, mad in rows:
        lines.append(f"  {v:<6} {ve_fp:>9.1f} {ve_q:>9.1f} {sp:>7.2f}x "
                     f"{snr:>8.2f} {lsd:>7.3f} {mad:>8.4f}")
    lines.append("  " + "-" * 60)
    lines.append(f"  Overall VE speedup: {overall_speedup:.2f}x   "
                 f"(fp32 {ve_fp_all.mean():.0f} ms -> int8 {ve_q_all.mean():.0f} ms)")
    lines.append(f"  Quality: SNR {snrs.mean():.2f} ± {snrs.std():.2f} dB   "
                 f"LSD {lsds.mean():.3f} ± {lsds.std():.3f}")
    lines.append("")
    lines.append("  Interpretation:")
    lines.append("    SNR > 30 dB  = near-transparent;  20-30 = minor;  < 20 = audible.")
    lines.append("    LSD lower is better (0 = identical spectra).")
    lines.append("    Audio saved per voice (fp32 vs int8) for A/B listening.")
    lines.append("=" * 76)
    report = "\n".join(lines)

    report_path = out_dir / "quant_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nReport: {report_path}\nAudio:  {audio_dir}/")


if __name__ == "__main__":
    main()
