"""Configuration and paths for the Supertonic benchmarking web tool."""
import os
from pathlib import Path

# Model / asset locations (match the benchmark scripts' defaults)
ONNX_DIR = os.environ.get("SUPERTONIC_ONNX_DIR", "/home/tdang1/.cache/supertonic3/onnx")
VOICE_DIR = os.environ.get("SUPERTONIC_VOICE_DIR", "/home/tdang1/.cache/supertonic3/voice_styles")

# All available voices (M4/M5/F4/F5 styles also exist on disk)
VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
DEFAULT_VOICE = "M1"

# Synthesis defaults
DEFAULT_STEPS = 6          # the recommended step count from the benchmark study
DEFAULT_SPEED = 1.05
DEFAULT_LANG = "en"
STEP_CHOICES = [4, 5, 6, 8, 12]

# ASR / WER
WHISPER_MODEL = os.environ.get("SUPERTONIC_WHISPER_MODEL", "base.en")
WHISPER_COMPUTE = "int8"

# Reading-failure heuristics (used in batch + interactive flags)
SILENCE_PEAK_THRESHOLD = 0.01     # |wav|.max() below this => near-silent
WER_FAILURE_THRESHOLD = 0.5       # absolute/relative WER above this => suspect (only where a ref exists)

# Output directory for batch runs (best-effort JSON persistence). Kept outside
# the package dir so it works regardless of package-dir ownership.
RUNS_DIR = Path(
    os.environ.get("SUPERTONIC_WEBBENCH_RUNS",
                   str(Path.home() / ".cache" / "supertonic_webbench_runs"))
)

# Single-flight: ONNX sessions are not safe for concurrent .run()
MAX_CONCURRENT_SYNTH = 1
