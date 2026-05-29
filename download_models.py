from supertonic import TTS
import os

tts = TTS(auto_download=True)
print("model_dir:", tts.model_dir)
print("sample_rate:", tts.sample_rate)

onnx_dir = tts.model_dir / "onnx"
if not onnx_dir.exists():
    onnx_dir = tts.model_dir
print("onnx_dir:", onnx_dir)
print("files:", list(onnx_dir.iterdir()))
