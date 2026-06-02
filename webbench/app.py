"""FastAPI app for the Supertonic TTS benchmarking web tool."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

import config
import corpus as corpus_mod
from audio import wav_bytes
from batch import run_batch
from jobs import registry
from models import ModelBundle
from pipeline_stream import synth_event_stream
from schemas import SynthRequest, CompareRequest, BatchRequest
from synth import run_one, stages_dict

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Supertonic models (once)...")
    app.state.bundle = ModelBundle()
    app.state.scorer = None  # lazy
    app.state.scorer_lock = asyncio.Lock()
    print(f"Models loaded. Voices: {app.state.bundle.voices}")
    yield


app = FastAPI(title="Supertonic TTS Benchmark", lifespan=lifespan)


async def get_scorer(app: FastAPI):
    """Lazily construct the Whisper scorer once."""
    if app.state.scorer is not None:
        return app.state.scorer
    async with app.state.scorer_lock:
        if app.state.scorer is None:
            print(f"Loading Whisper scorer ({config.WHISPER_MODEL})...")
            from wer import WhisperScorer
            loop = asyncio.get_running_loop()
            app.state.scorer = await loop.run_in_executor(None, WhisperScorer)
            print("Whisper scorer ready.")
    return app.state.scorer


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@app.get("/api/voices")
async def voices():
    b = app.state.bundle
    return {"voices": b.voices, "default": config.DEFAULT_VOICE,
            "step_choices": config.STEP_CHOICES, "default_steps": config.DEFAULT_STEPS,
            "default_speed": config.DEFAULT_SPEED}


@app.get("/api/corpus")
async def corpus_endpoint():
    return corpus_mod.category_payload()


@app.post("/api/synth")
async def synth(req: SynthRequest):
    b = app.state.bundle
    if req.voice not in b.voice_styles:
        raise HTTPException(400, f"unknown voice {req.voice!r}")

    def scorer_getter():
        # synchronous accessor used inside the stream; ensures scorer exists
        return app.state.scorer

    if req.run_wer:
        await get_scorer(app)  # ensure loaded before streaming WER

    return EventSourceResponse(synth_event_stream(b, scorer_getter, req))


@app.get("/api/audio/{audio_id}")
async def audio(audio_id: str):
    b = app.state.bundle
    wav = b.get_audio(audio_id)
    if wav is None:
        raise HTTPException(404, "audio expired or not found")
    data = wav_bytes(wav, b.sample_rate)
    return Response(content=data, media_type="audio/wav")


@app.post("/api/compare")
async def compare(req: CompareRequest):
    b = app.state.bundle
    loop = asyncio.get_running_loop()
    scorer = await get_scorer(app) if req.run_wer else None

    results = []
    transcripts = []
    for cfg in req.configs:
        if cfg.voice not in b.voice_styles:
            raise HTTPException(400, f"unknown voice {cfg.voice!r}")
        wav, t = await loop.run_in_executor(
            None, run_one, b, req.text, cfg.voice, cfg.steps, cfg.speed, req.lang, None, req.seed)
        audio_id = b.store_audio(wav)
        entry = stages_dict(t)
        entry["label"] = cfg.label
        entry["audio_url"] = f"/api/audio/{audio_id}"
        if scorer is not None:
            transcript, asr_ms = await loop.run_in_executor(
                None, scorer.transcribe, wav, b.sample_rate)
            entry["transcript"] = transcript
            entry["asr_ms"] = asr_ms
            transcripts.append(transcript)
        results.append(entry)

    relative = None
    if scorer is not None and len(transcripts) >= 2:
        base = transcripts[0]
        relative = []
        for i in range(1, len(transcripts)):
            sc = scorer.score(base, transcripts[i])
            relative.append({"from": req.configs[0].label, "to": req.configs[i].label,
                             "wer": sc["wer"], "cer": sc["cer"]})
    return {"results": results, "relative": relative}


@app.post("/api/batch")
async def batch_start(req: BatchRequest):
    b = app.state.bundle
    cases = corpus_mod.cases_for(req.categories)
    valid_voices = [v for v in req.voices if v in b.voice_styles]
    total = len(cases) * len(valid_voices) * len(req.steps) * req.reps
    if total == 0:
        raise HTTPException(400, "empty batch (check categories/voices/steps)")
    job = registry.create(total, req.model_dump())

    scorer = await get_scorer(app) if req.run_wer else None
    loop = asyncio.get_running_loop()

    def runner():
        try:
            run_batch(b, scorer, job, req)
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = repr(e)

    loop.run_in_executor(None, runner)
    return {"job_id": job.id, "total": total}


@app.get("/api/batch/{job_id}")
async def batch_status(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.snapshot()


@app.get("/api/batch/{job_id}/events")
async def batch_events(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    async def gen():
        last = -1
        while True:
            snap = job.snapshot()
            if snap["done"] != last:
                last = snap["done"]
                yield {"event": "progress", "data": _json(snap)}
            if job.status == "complete":
                yield {"event": "complete", "data": _json({"result_url": f"/api/batch/{job_id}/result"})}
                return
            if job.status == "error":
                yield {"event": "error", "data": _json({"error": job.error})}
                return
            await asyncio.sleep(0.4)

    return EventSourceResponse(gen())


@app.get("/api/batch/{job_id}/result")
async def batch_result(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.result is None:
        raise HTTPException(409, f"job not complete (status={job.status})")
    return JSONResponse(job.result)


@app.post("/api/warmup")
async def warmup():
    await get_scorer(app)
    return {"ok": True}


def _json(obj):
    import json
    return json.dumps(obj)


# Static frontend (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
