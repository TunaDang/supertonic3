"""ModelBundle: load the Supertonic ONNX pipeline once and reuse it.

Reuses the existing repo code in py/helper.py and step_benchmark.py rather than
re-implementing inference. The Whisper ASR scorer is loaded lazily (see wer.py).
"""
import os
import sys
import threading
import uuid

import numpy as np
import onnxruntime as ort

# Make the repo's pipeline importable (step_benchmark adds its own py/ dir).
SUPERTONIC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SUPERTONIC_ROOT not in sys.path:
    sys.path.insert(0, SUPERTONIC_ROOT)

from helper import (  # noqa: E402  (path set above)
    load_cfgs,
    load_onnx_all,
    load_text_processor,
    load_voice_style,
)

import config  # noqa: E402


class ModelBundle:
    """Holds the loaded ONNX sessions, text processor, voice styles, and a
    short-lived in-memory audio store for browser playback."""

    def __init__(self, onnx_dir: str = config.ONNX_DIR, voice_dir: str = config.VOICE_DIR):
        self.onnx_dir = onnx_dir
        self.voice_dir = voice_dir
        opts = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]

        self.cfgs = load_cfgs(onnx_dir)
        self.sample_rate = self.cfgs["ae"]["sample_rate"]
        (self.dp, self.text_enc, self.vector_est, self.vocoder) = load_onnx_all(
            onnx_dir, opts, providers
        )
        self.text_processor = load_text_processor(onnx_dir)

        # Preload every voice style that exists on disk.
        self.voice_styles = {}
        for v in config.VOICES:
            path = os.path.join(voice_dir, f"{v}.json")
            if os.path.exists(path):
                self.voice_styles[v] = load_voice_style([path])
        self.voices = list(self.voice_styles.keys())

        # id -> float32 wav [1, N]; capped FIFO so memory stays bounded.
        self._audio = {}
        self._audio_order = []
        self._audio_lock = threading.Lock()
        self._audio_cap = 64

    def voice_style(self, voice: str):
        if voice not in self.voice_styles:
            raise KeyError(f"unknown voice {voice!r}; available: {self.voices}")
        return self.voice_styles[voice]

    def store_audio(self, wav: np.ndarray) -> str:
        audio_id = uuid.uuid4().hex[:16]
        with self._audio_lock:
            self._audio[audio_id] = wav
            self._audio_order.append(audio_id)
            while len(self._audio_order) > self._audio_cap:
                old = self._audio_order.pop(0)
                self._audio.pop(old, None)
        return audio_id

    def get_audio(self, audio_id: str):
        with self._audio_lock:
            return self._audio.get(audio_id)
