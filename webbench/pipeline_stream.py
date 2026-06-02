"""SSE bridge: run the (synchronous, CPU-bound) pipeline in a threadpool and
stream per-stage timing events to the browser as they complete.

A single asyncio.Semaphore serializes all synthesis — ONNX sessions are not
safe for concurrent .run(), and serialization also keeps timings clean.
"""
import asyncio
import json

import numpy as np

import config
from synth import run_one, stages_dict

_synth_sem = asyncio.Semaphore(config.MAX_CONCURRENT_SYNTH)


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data)}


async def synth_event_stream(bundle, scorer_getter, req):
    """Async generator of SSE events for POST /api/synth.

    scorer_getter() -> WhisperScorer (lazy) or None if WER not requested.
    """
    loop = asyncio.get_running_loop()
    aq: asyncio.Queue = asyncio.Queue()

    def on_stage(name, ms, extra):  # called from worker thread
        payload = {"type": name, "ms": ms, **extra}
        loop.call_soon_threadsafe(aq.put_nowait, payload)

    def worker():
        try:
            wav, t = run_one(bundle, req.text, req.voice, req.steps,
                             req.speed, req.lang, on_stage=on_stage)
            audio_id = bundle.store_audio(wav)
            result = stages_dict(t)
            result["audio_id"] = audio_id
            peak = float(np.abs(wav).max()) if wav.size else 0.0
            result["peak"] = peak
            loop.call_soon_threadsafe(aq.put_nowait, {"type": "_done", "result": result, "wav_id": audio_id})
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(aq.put_nowait, {"type": "_error", "message": repr(e)})
        finally:
            loop.call_soon_threadsafe(aq.put_nowait, {"type": "_end"})

    async with _synth_sem:
        loop.run_in_executor(None, worker)

        done_result = None
        while True:
            item = await aq.get()
            typ = item["type"]
            if typ == "_end":
                break
            if typ == "_error":
                yield _sse("error", {"message": item["message"]})
                return
            if typ == "_done":
                done_result = item["result"]
                yield _sse("done", {
                    "audio_url": f"/api/audio/{done_result['audio_id']}",
                    "stages": done_result["stages"],
                    "ve_per_step_ms": done_result["ve_per_step_ms"],
                    "total_ms": done_result["total_ms"],
                    "audio_duration_s": done_result["audio_duration_s"],
                    "rtf": done_result["rtf"],
                    "peak": done_result["peak"],
                })
            else:
                # per-stage (and ve_step) events
                yield _sse("stage", item)

    # WER runs AFTER synthesis completes, outside the synth semaphore so its
    # latency is never folded into synth timing.
    if req.run_wer and done_result is not None:
        scorer = scorer_getter()
        wav = bundle.get_audio(done_result["audio_id"])
        if scorer is None or wav is None:
            yield _sse("wer", {"error": "scorer or audio unavailable"})
            return
        try:
            transcript, asr_ms = await loop.run_in_executor(
                None, scorer.transcribe, wav, bundle.sample_rate)
            if req.reference:  # absolute WER (clean prose)
                scores = scorer.score(req.reference, transcript)
                mode = "absolute"
                reference = req.reference
            else:  # self-reference: WER of ASR vs the input text (informational)
                scores = scorer.score(req.text, transcript)
                mode = "vs_input"
                reference = req.text
            yield _sse("wer", {
                "transcript": transcript,
                "reference": reference,
                "mode": mode,
                "wer": scores["wer"],
                "cer": scores["cer"],
                "asr_ms": asr_ms,
                "asr_model": scorer.model_size,
            })
        except Exception as e:  # noqa: BLE001
            yield _sse("wer", {"error": repr(e)})
