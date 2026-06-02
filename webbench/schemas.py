"""Pydantic request/response models for the benchmarking API."""
from typing import List, Optional

from pydantic import BaseModel, Field

import config


class SynthRequest(BaseModel):
    text: str
    voice: str = config.DEFAULT_VOICE
    steps: int = config.DEFAULT_STEPS
    speed: float = config.DEFAULT_SPEED
    lang: str = config.DEFAULT_LANG
    run_wer: bool = False
    reference: Optional[str] = None  # absolute-WER target (clean-prose cases)
    seed: Optional[int] = 0          # fixed => reproducible; null => random take
    verbalize_input: bool = False    # pre-verbalize text before synthesis
    verbalize_ref: bool = False      # verbalize the WER reference before scoring


class ConfigSpec(BaseModel):
    label: str
    voice: str = config.DEFAULT_VOICE
    steps: int = config.DEFAULT_STEPS
    speed: float = config.DEFAULT_SPEED
    verbalize_input: bool = False  # per-config: A/B verbalized vs raw


class CompareRequest(BaseModel):
    text: str
    lang: str = config.DEFAULT_LANG
    run_wer: bool = True
    seed: Optional[int] = 0  # shared across configs so only the config differs
    configs: List[ConfigSpec] = Field(..., min_length=2)


class BatchRequest(BaseModel):
    categories: List[str] = Field(default_factory=list)  # empty => all
    voices: List[str] = Field(default_factory=lambda: ["M1", "F1"])
    steps: List[int] = Field(default_factory=lambda: [config.DEFAULT_STEPS])
    speed: float = config.DEFAULT_SPEED
    lang: str = config.DEFAULT_LANG
    reps: int = 1
    run_wer: bool = True
    seed: Optional[int] = 0  # fixed => reproducible WER across the suite
    verbalize_input: bool = False
    verbalize_ref: bool = False
