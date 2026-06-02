"""ASR round-trip WER/CER scoring via faster-whisper + jiwer.

Quality methodology: absolute WER is only meaningful on clean prose (where the
reference is the exact text). For edge cases use RELATIVE WER — transcribe two
configs of the SAME text and compare their transcripts — which cancels the
systematic ASR bias. ASR latency is reported separately from synth latency.
"""
import re
import time

import jiwer
import numpy as np
from scipy.signal import resample_poly

import config


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Applied to BOTH ref
    and hyp before scoring (avoids jiwer-version transform-kwarg churn)."""
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class WhisperScorer:
    def __init__(self, model_size: str = config.WHISPER_MODEL,
                 compute_type: str = config.WHISPER_COMPUTE):
        from faster_whisper import WhisperModel
        # ~140MB download on first construction; HF-cached afterwards.
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        self.model_size = model_size

    def transcribe(self, wav_44k: np.ndarray, sample_rate: int = 44100):
        """Returns (transcript, asr_ms). Resamples 44.1k -> 16k for Whisper."""
        mono = np.asarray(wav_44k).squeeze().astype(np.float32)
        if sample_rate != 16000:
            # 44100/16000 reduces to 441/160
            from math import gcd
            g = gcd(sample_rate, 16000)
            mono = resample_poly(mono, 16000 // g, sample_rate // g).astype(np.float32)
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(mono, language="en", beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        asr_ms = (time.perf_counter() - t0) * 1000.0
        return text, asr_ms

    @staticmethod
    def score(reference: str, hypothesis: str) -> dict:
        """WER + CER with identical normalization on both sides."""
        ref = _normalize(reference)
        hyp = _normalize(hypothesis)
        if not ref:
            return {"wer": None, "cer": None}
        try:
            wer = float(jiwer.wer(ref, hyp))
        except Exception:
            wer = None
        try:
            cer = float(jiwer.cer(ref, hyp))
        except Exception:
            cer = None
        return {"wer": wer, "cer": cer}
