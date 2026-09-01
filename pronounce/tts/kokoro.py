"""Kokoro 的 G2P（misaki + spaCy）和 TTS，权重来自本机 ``llm/``。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pronounce.common.espeak import ensure_espeak
from pronounce.common.spacy_path import activate as activate_spacy
from pronounce.paths import kokoro_model

KOKORO_SAMPLE_RATE = 24_000
DEFAULT_VOICE = "af_heart"
# Kokoro 内部口音码：美音 a、英音 b。也接受已经是 a/b 的值。
_LANG = {"en-us": "a", "en-gb": "b", "a": "a", "b": "b"}

_kmodel = None
_pipelines: dict[str, object] = {}


def load_tts(device: str = "cpu"):
    """Load Kokoro once per process. Safe to call repeatedly."""
    global _kmodel
    from kokoro import KModel

    if _kmodel is None:
        activate_spacy()
        ensure_espeak()
        root = _root()
        _kmodel = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(root / "config.json"),
            model=str(root / "kokoro-v1_0.pth"),
        ).to(device).eval()
    return _kmodel


def lang_code(lang: str) -> str:
    """把 ``en-us`` / ``en-gb`` 收成 Kokoro 的 ``a`` / ``b``。"""
    code = _LANG.get((lang or "en-us").lower())
    if code is None:
        raise ValueError(f"unknown lang {lang!r}; use en-us or en-gb")
    return code


def _root() -> Path:
    """Kokoro 快照根；缺 config.json 或 .pth 则报错（运行时不下载）。"""
    root = kokoro_model()
    if not (root / "config.json").is_file() or not (root / "kokoro-v1_0.pth").is_file():
        raise FileNotFoundError(f"Kokoro snapshot missing or incomplete at {root}")
    return root


def voice_path(voice: str) -> Path:
    """解析音色：可以是已有的 ``.pt`` 文件路径，或 ``voices/<id>.pt`` 里的 id。"""
    root = _root()
    path = Path(voice)
    if path.suffix == ".pt" and path.is_file():
        return path.resolve()
    pt = root / "voices" / f"{voice}.pt"
    if not pt.is_file():
        raise FileNotFoundError(f"Kokoro voice not found: {pt}")
    return pt


def phonemize(text: str, lang: str = "en-us") -> str:
    """字素→音素：misaki（spaCy 管道 + espeak 兜底）。"""
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
    """合成 *text*；单声道 float32，24 kHz。

    Kokoro 按句切块生成，这里把各块 ``np.concatenate`` 接成一条波形。
    ``KModel`` 经 ``load_tts`` 进程内只建一次；``KPipeline`` 按口音码缓存。
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    from kokoro import KPipeline

    model = load_tts(device)
    pt = voice_path(voice)
    code = lang_code(lang)
    pipeline = _pipelines.get(code)
    if pipeline is None:
        pipeline = KPipeline(
            lang_code=code,
            repo_id="hexgrad/Kokoro-82M",
            model=model,
        )
        _pipelines[code] = pipeline
    chunks = []
    for _, _, audio in pipeline(text, voice=str(pt), model=model):
        if audio is not None and len(audio) > 0:
            chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def listen_sample_rate(speed: float, native: int = KOKORO_SAMPLE_RATE) -> int:
    """播放用的 wav 采样率 = 原生速率 × speed（mimora 的「磁带减速」：慢放就降低采样率）。"""
    if speed <= 0 or speed > 2:
        raise ValueError(f"speed must be in (0, 2], got {speed}")
    return max(1, int(round(native * speed)))
