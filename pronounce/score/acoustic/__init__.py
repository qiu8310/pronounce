# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Mimora 声学发音分析包。

把 OpenPronounce（MIT）的声学对比核当库来用。唯一入口是 ``analyze``；
``load_models`` / ``warm_up`` 管 Wav2Vec2 生命周期（模式启动时在后台线程调用）。

本包不依赖 GUI / 宿主：设置来自自己的 ``AnalyzerConfig``。宿主启动时
``configure()`` 注入一次；不注入则用内置默认，包可以单独跑。
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
