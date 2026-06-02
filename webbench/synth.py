"""Thin wrappers around the repo's run_instrumented for the web tool."""
import os
import sys
import threading

import numpy as np

# Global single-flight lock: ONNX sessions are not safe for concurrent .run(),
# and serialization keeps timings clean. Every synthesis path acquires this.
SYNTH_LOCK = threading.Lock()

# Ensure the repo pipeline is importable (also done in models.py).
SUPERTONIC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SUPERTONIC_ROOT not in sys.path:
    sys.path.insert(0, SUPERTONIC_ROOT)

from step_benchmark import run_instrumented  # noqa: E402

STAGE_ORDER = [
    ("preprocess", "preprocess_ms"),
    ("duration_predictor", "duration_predictor_ms"),
    ("text_encoder", "text_encoder_ms"),
    ("noisy_latent", "noisy_latent_ms"),
    ("vector_estimator", "vector_estimator_total_ms"),
    ("vocoder", "vocoder_ms"),
    ("audio_trim", "audio_trim_ms"),
]


def stages_dict(t) -> dict:
    """StepTimings -> JSON-friendly per-stage breakdown."""
    stages = {name: getattr(t, attr) for name, attr in STAGE_ORDER}
    rtf = (t.total_ms / 1000.0) / t.audio_duration_s if t.audio_duration_s else None
    return {
        "stages": stages,
        "ve_per_step_ms": list(t.vector_estimator_per_step_ms),
        "total_ms": t.total_ms,
        "audio_duration_s": t.audio_duration_s,
        "rtf": rtf,
        "total_steps": t.total_steps,
    }


def run_one(bundle, text, voice, steps, speed, lang, on_stage=None, seed=None):
    """Run one synthesis; returns (wav, StepTimings). Single-flight.

    If seed is not None, the flow-matching noise is seeded so the output (and
    thus the WER) is reproducible. The only stochastic step in the pipeline is
    the noisy-latent sampling; seeding the global RNG just before the run (under
    the lock) makes it deterministic. seed=None keeps the original random "take".
    """
    style = bundle.voice_style(voice)
    with SYNTH_LOCK:
        if seed is not None:
            np.random.seed(seed)
        return run_instrumented(
            bundle.cfgs, bundle.text_processor,
            bundle.dp, bundle.text_enc, bundle.vector_est, bundle.vocoder,
            text, lang, style, steps, voice, "web",
            speed=speed, verbose=False, on_stage=on_stage,
        )
