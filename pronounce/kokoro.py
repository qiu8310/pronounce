"""Kokoro G2P (misaki + spaCy) and TTS from local llm/ weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pronounce.common.espeak import ensure_espeak
from pronounce.common.spacy_path import activate as activate_spacy
from pronounce.paths import kokoro_model

KOKORO_SAMPLE_RATE = 24_000
DEFAULT_VOICE = "af_heart"
_LANG = {"en-us": "a", "en-gb": "b", "a": "a", "b": "b"}


def lang_code(lang: str) -> str:
    code = _LANG.get((lang or "en-us").lower())
    if code is None:
        raise ValueError(f"unknown lang {lang!r}; use en-us or en-gb")
    return code


def _root() -> Path:
    root = kokoro_model()
    if not (root / "config.json").is_file() or not (root / "kokoro-v1_0.pth").is_file():
        raise FileNotFoundError(f"Kokoro snapshot missing or incomplete at {root}")
    return root


def voice_path(voice: str) -> Path:
    root = _root()
    path = Path(voice)
    if path.suffix == ".pt" and path.is_file():
        return path.resolve()
    pt = root / "voices" / f"{voice}.pt"
    if not pt.is_file():
        raise FileNotFoundError(f"Kokoro voice not found: {pt}")
    return pt


def phonemize(text: str, lang: str = "en-us") -> str:
    """Grapheme-to-phoneme via misaki (spaCy pipeline + espeak fallback)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    activate_spacy()
    ensure_espeak()
    from misaki import en, espeak

    british = lang_code(lang) == "b"
    try:
        fallback = espeak.EspeakFallback(british=british)
    except Exception:
        fallback = None
    g2p = en.G2P(trf=False, british=british, fallback=fallback, unk="")
    phonemes, _tokens = g2p(text)
    return (phonemes or "").strip()


def synthesize(text: str, voice: str = DEFAULT_VOICE, lang: str = "en-us",
               device: str = "cpu") -> np.ndarray:
    """Synthesize *text*; mono float32 at 24 kHz."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    activate_spacy()
    ensure_espeak()
    from kokoro import KModel, KPipeline

    root = _root()
    pt = voice_path(voice)
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(root / "config.json"),
        model=str(root / "kokoro-v1_0.pth"),
    ).to(device).eval()
    pipeline = KPipeline(
        lang_code=lang_code(lang),
        repo_id="hexgrad/Kokoro-82M",
        model=model,
    )
    chunks = []
    for _, _, audio in pipeline(text, voice=str(pt), model=model):
        if audio is not None and len(audio) > 0:
            chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def listen_sample_rate(speed: float, native: int = KOKORO_SAMPLE_RATE) -> int:
    """Wav sample rate that plays *speed* times native tempo (mimora tape-slow)."""
    if speed <= 0 or speed > 2:
        raise ValueError(f"speed must be in (0, 2], got {speed}")
    return max(1, int(round(native * speed)))
