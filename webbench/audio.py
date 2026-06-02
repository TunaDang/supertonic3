"""Encode float32 [1, N] @ 44.1kHz to 16-bit PCM WAV bytes for the browser."""
import io

import numpy as np
import soundfile as sf


def wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    """float32 array (any shape squeezable to 1-D) -> 16-bit PCM WAV bytes."""
    mono = np.asarray(wav).squeeze().astype(np.float32)
    mono = np.clip(mono, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, mono, int(sample_rate), subtype="PCM_16", format="WAV")
    return buf.getvalue()
