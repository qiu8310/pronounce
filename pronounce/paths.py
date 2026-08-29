"""解析本机模型根目录 ``MODELS_HOME``，以及各能力对应的权重路径。

约定：权重一律在 ``$MODELS_HOME/llm/<family>/...``，运行时不下载。
"""

from __future__ import annotations

import os
from pathlib import Path


def models_home() -> Path:
    """共享模型根目录。

    优先读环境变量 ``MODELS_HOME``；未设置时，若本仓库里有 ``llm/`` 就用仓库根。
    expanduser() 会把 ``~`` 展开成家目录；resolve() 变成绝对路径并解析符号链接。
    """
    raw = os.environ.get("MODELS_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    # __file__ 是 pronounce/paths.py；parents[2] 再上两级到 models 仓库根。
    inferred = Path(__file__).resolve().parents[2]
    if (inferred / "llm").is_dir():
        return inferred
    raise RuntimeError(
        "MODELS_HOME is not set; export it to the shared models directory"
    )


def wav2vec2_model(name: str) -> Path:
    """Wav2Vec2 快照目录，例如 ``wav2vec2-xlsr-53-espeak-cv-ft``。"""
    return models_home() / "llm" / "wav2vec2" / name


def kokoro_model() -> Path:
    """Kokoro-82M TTS 快照目录。"""
    return models_home() / "llm" / "kokoro" / "Kokoro-82M"


def spacy_dir() -> Path:
    """解压好的 spaCy 管道目录（Kokoro G2P 用的 en_core_web_sm）。"""
    return models_home() / "llm" / "spacy"


def melo_chinese() -> Path:
    """MeloTTS 中文 checkpoint 目录（config.json + checkpoint.pth）。"""
    return models_home() / "llm" / "melo" / "MeloTTS-Chinese"


def melo_bert() -> Path:
    """MeloTTS 中文 BERT 特征目录（hfl/chinese-roberta-wwm-ext-large）。"""
    return models_home() / "llm" / "melo" / "chinese-roberta-wwm-ext-large"
