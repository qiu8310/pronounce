# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Mimora 音素发音引擎（按文本打分，不必有参考录音）。

相对声学核更轻：用短语**文本**（espeak 参考音素）加上用户音频的 wav2vec2
音素 ASR 来打分——不需要每句都有参考录音。

公开 API 与声学包对齐，宿主用同一套调用切换引擎：``analyze`` 是唯一入口；
``load_models`` / ``warm_up`` 管识别器生命周期。设置来自本包 ``AnalyzerConfig``；
宿主 ``configure()`` 注入一次，默认值保证能独立跑。

本包不碰 GUI；返回的结果结构和 ``pronounce.score.acoustic.PronunciationResult``
相同，UI 对引擎保持中立。
"""

from .config import AnalyzerConfig, configure, get_config
from .speech import (
    PronunciationResult,
    analyze,
    load_models,
    warm_up,
)

__all__ = [
    "analyze",
    "load_models",
    "warm_up",
    "PronunciationResult",
    "AnalyzerConfig",
    "configure",
    "get_config",
]
