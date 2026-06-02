"""Batch suite runner + aggregation for the dashboard.

Runs corpus cases x voices x steps x reps, collects per-stage timings, optionally
scores WER (absolute on clean-prose refs; vs-input proxy elsewhere), flags
reading failures, and aggregates to latency percentiles + per-category metrics.
"""
import json

import numpy as np

import config
import corpus as corpus_mod
from synth import run_one, STAGE_ORDER


def _pct(values, p):
    return float(np.percentile(values, p)) if values else None


def _summary(values):
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "n": int(a.size),
    }


def run_batch(bundle, scorer, job, req):
    """Execute the suite (blocking; call in a threadpool). Updates `job` and
    sets job.result on completion."""
    cases = corpus_mod.cases_for(req.categories)
    rows = []
    failures = []

    for case in cases:
        for voice in req.voices:
            if voice not in bundle.voice_styles:
                continue
            for steps in req.steps:
                for rep in range(req.reps):
                    job.current = {"case_id": case.id, "voice": voice, "steps": steps, "rep": rep}
                    try:
                        wav, t = run_one(bundle, case.text, voice, steps, req.speed,
                                         req.lang, seed=req.seed)
                    except Exception as e:  # noqa: BLE001
                        failures.append({"case_id": case.id, "voice": voice, "steps": steps,
                                         "rep": rep, "reason": f"exception: {e!r}",
                                         "wer": None, "peak": None})
                        job.done += 1
                        continue

                    peak = float(np.abs(wav).max()) if wav.size else 0.0
                    row = {
                        "case_id": case.id, "category": case.category, "voice": voice,
                        "steps": steps, "rep": rep, "peak": peak,
                        "total_ms": t.total_ms, "rtf": (t.total_ms / 1000.0) / t.audio_duration_s
                        if t.audio_duration_s else None,
                        "stages": {name: getattr(t, attr) for name, attr in STAGE_ORDER},
                        "wer": None, "cer": None, "wer_mode": None,
                    }

                    # WER (only on first rep to save ASR time)
                    if req.run_wer and scorer is not None and rep == 0:
                        try:
                            transcript, _asr_ms = scorer.transcribe(wav, bundle.sample_rate)
                            if case.reference:
                                sc = scorer.score(case.reference, transcript)
                                row["wer_mode"] = "absolute"
                            else:
                                sc = scorer.score(case.text, transcript)
                                row["wer_mode"] = "vs_input"
                            row["wer"] = sc["wer"]
                            row["cer"] = sc["cer"]
                            row["transcript"] = transcript
                        except Exception:  # noqa: BLE001
                            pass

                    # Reading-failure heuristics
                    reasons = []
                    if peak < config.SILENCE_PEAK_THRESHOLD:
                        reasons.append(f"near-silent (peak={peak:.4f})")
                    # high WER only trustworthy where there's a clean reference
                    if case.reference and row["wer"] is not None and row["wer"] > config.WER_FAILURE_THRESHOLD:
                        reasons.append(f"high WER ({row['wer']:.2f})")
                    if reasons:
                        failures.append({"case_id": case.id, "voice": voice, "steps": steps,
                                         "rep": rep, "reason": "; ".join(reasons),
                                         "wer": row["wer"], "peak": peak})

                    rows.append(row)
                    job.done += 1

    job.result = _aggregate(rows, failures, req)
    job.status = "complete"
    _persist(job)


def _aggregate(rows, failures, req):
    per_stage = {}
    for name, _attr in STAGE_ORDER:
        per_stage[name] = _summary([r["stages"][name] for r in rows])
    per_stage["total"] = _summary([r["total_ms"] for r in rows])
    rtf_vals = [r["rtf"] for r in rows if r["rtf"] is not None]

    # per-category
    per_category = {}
    cats = sorted({r["category"] for r in rows})
    for cat in cats:
        crows = [r for r in rows if r["category"] == cat]
        wer_vals = [r["wer"] for r in crows if r["wer"] is not None]
        cer_vals = [r["cer"] for r in crows if r["cer"] is not None]
        modes = {r["wer_mode"] for r in crows if r["wer_mode"]}
        per_category[cat] = {
            "n": len(crows),
            "total_ms_p50": _pct([r["total_ms"] for r in crows], 50),
            "rtf_p50": _pct([r["rtf"] for r in crows if r["rtf"] is not None], 50),
            "wer_mean": float(np.mean(wer_vals)) if wer_vals else None,
            "cer_mean": float(np.mean(cer_vals)) if cer_vals else None,
            "wer_mode": "absolute" if "absolute" in modes else ("vs_input" if modes else None),
        }

    # per-voice
    per_voice = {}
    for v in sorted({r["voice"] for r in rows}):
        vrows = [r for r in rows if r["voice"] == v]
        per_voice[v] = {
            "total_ms_p50": _pct([r["total_ms"] for r in vrows], 50),
            "rtf_p50": _pct([r["rtf"] for r in vrows if r["rtf"] is not None], 50),
            "n": len(vrows),
        }

    return {
        "per_stage": per_stage,
        "rtf": _summary(rtf_vals),
        "per_category": per_category,
        "per_voice": per_voice,
        "reading_failures": failures,
        "raw_count": len(rows),
        "config": req.model_dump() if hasattr(req, "model_dump") else dict(req),
    }


def _persist(job):
    try:
        out_dir = config.RUNS_DIR / job.id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(job.result, f, indent=2)
    except Exception:  # noqa: BLE001
        pass
