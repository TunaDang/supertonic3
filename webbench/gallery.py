"""On-demand, regenerable sample gallery.

Synthesizes a curated phrase set at a fixed seed and caches the audio + a JSON
manifest under config.RUNS_DIR/gallery (gitignored, regenerable). Highlights
expression tags by synthesizing WITH vs WITHOUT the tag and reporting the
duration delta, across a couple of voices, so their (subtle) audibility is
observable. Audio is NOT committed to git.
"""
import json
import re

import numpy as np

import config
import corpus as corpus_mod
from audio import wav_bytes
from synth import run_one

GALLERY_DIR = config.RUNS_DIR / "gallery"

# A representative sample across categories (synthesized as-is, M1).
SAMPLE_IDS = ["harvard_01", "money_01", "telephone_01", "measure_01",
              "homograph_02", "prosody_01", "cardinal_01", "date_02"]
EXPRESSION_VOICES = ["M1", "F1"]
_TAG_RE = re.compile(r"\s*<[^>]+>\s*")


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text)).strip()


def gallery_total(spec) -> int:
    voice = spec.get("voice", "M1")
    steps = spec.get("steps", config.DEFAULT_STEPS)
    expr = [c for c in corpus_mod.CASES if c.category == "expression"]
    return len(SAMPLE_IDS) + len(expr) * len(EXPRESSION_VOICES) * 2


def build_gallery(bundle, scorer, job, spec):
    """Blocking; run in a threadpool. Updates job.done; writes manifest + WAVs."""
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    seed = spec.get("seed", 0)
    steps = spec.get("steps", config.DEFAULT_STEPS)
    voice = spec.get("voice", "M1")
    by_id = {c.id: c for c in corpus_mod.CASES}

    def _synth_to_file(text, vc, fname):
        wav, t = run_one(bundle, text, vc, steps, config.DEFAULT_SPEED, "en", seed=seed)
        with open(GALLERY_DIR / fname, "wb") as f:
            f.write(wav_bytes(wav, bundle.sample_rate))
        peak = float(np.abs(wav).max()) if wav.size else 0.0
        return t, peak, wav

    items = []
    for cid in SAMPLE_IDS:
        case = by_id.get(cid)
        if case is None or voice not in bundle.voice_styles:
            job.done += 1
            continue
        job.current = {"section": "samples", "id": cid, "voice": voice}
        fname = f"sample_{cid}_{voice}.wav"
        t, peak, wav = _synth_to_file(case.text, voice, fname)
        row = {"id": cid, "category": case.category, "text": case.text, "voice": voice,
               "steps": steps, "file": fname, "duration_s": t.audio_duration_s,
               "total_ms": t.total_ms, "peak": peak,
               "wer": None, "cer": None, "transcript": None}
        if scorer is not None:
            transcript, _ = scorer.transcribe(wav, bundle.sample_rate)
            # Uniform: always score ASR vs the input text. For clean prose this
            # is a true intelligibility WER; for number/symbol cases it is
            # confounded (use the Interactive Verbalize toggle for a fair number).
            sc = scorer.score(case.text, transcript)
            row.update(wer=sc["wer"], cer=sc["cer"], transcript=transcript)
        items.append(row)
        job.done += 1

    # Expression with/without pairs (the headline of the gallery)
    expr_pairs = []
    for case in [c for c in corpus_mod.CASES if c.category == "expression"]:
        tag_m = re.search(r"<[^>]+>", case.text)
        tag = tag_m.group(0) if tag_m else "<tag>"
        without_text = _strip_tags(case.text)
        for vc in EXPRESSION_VOICES:
            if vc not in bundle.voice_styles:
                job.done += 2
                continue
            job.current = {"section": "expression", "id": case.id, "voice": vc}
            fw = f"expr_{case.id}_{vc}_with.wav"
            fo = f"expr_{case.id}_{vc}_without.wav"
            tw, _, _ = _synth_to_file(case.text, vc, fw)
            job.done += 1
            to, _, _ = _synth_to_file(without_text, vc, fo)
            job.done += 1
            expr_pairs.append({
                "id": case.id, "tag": tag, "voice": vc, "steps": steps,
                "with_text": case.text, "without_text": without_text,
                "with_file": fw, "without_file": fo,
                "with_duration_s": tw.audio_duration_s,
                "without_duration_s": to.audio_duration_s,
                "delta_ms": (tw.audio_duration_s - to.audio_duration_s) * 1000.0,
            })

    manifest = {"built": True, "seed": seed, "steps": steps, "voice": voice,
                "items": items, "expression_pairs": expr_pairs}
    with open(GALLERY_DIR / "gallery.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    job.result = manifest
    job.status = "complete"


def load_manifest():
    path = GALLERY_DIR / "gallery.json"
    if not path.exists():
        return {"built": False}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
