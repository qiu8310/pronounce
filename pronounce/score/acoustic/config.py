# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""声学分析库的配置。

``pronunciation/acoustic/`` 是与 GUI、宿主无关的核（改编自 OpenPronounce）。
它不能回头 import 宿主，所以每个可调项都放在下面这个小的 :class:`AnalyzerConfig` 里。

库自带能用的默认值，完全可独立运行：import 后直接调
:func:`pronounce.score.acoustic.analyze` 即可。宿主在启动时用 :func:`configure`
注入自己的值，类似 ``logging.basicConfig``——之后分析只读这里当前生效的配置。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalyzerConfig:
    """发音分析器读的设置。

    frozen=True：实例创建后字段不能改（不可变），避免分析中途被偷偷改配置。
    要换设置就再 ``configure()`` 一个新实例。

    默认值让库能单独用；宿主构造一个 ``AnalyzerConfig`` 再交给 :func:`configure`。
    """

    # Wav2Vec2 权重和运行设备（"cuda"/"cpu"）。
    model_name: str = "facebook/wav2vec2-large-960h"
    device: str = "cpu"
    # 把参考文本转成音素时用的 espeak 口音（"en-us"/"en-gb"）。
    espeak_language: str = "en-us"
    # 0-100 分达到或超过此值视为合格。
    score_threshold: float = 70.0
    # 校准前的声学地板：一次「读得不错」的典型逐步余弦 DTW 距离。
    # calibration.json 里的按用户值会覆盖它。
    acoustic_good: float = 0.20
    # 校准样本日志（acoustic_samples.jsonl）写到哪个目录。
    log_dir: Path = Path("logs")
    # 当前练习用户；校准按人分（未设则为 ""）。
    user_name: str = ""
    # 本机用户校准文件的读写路径。None 时库自己找包旁边的 calibration.json
    # （独立使用和评估工具期望的位置）。宿主应注入自己的路径：那是机器本地
    # 状态，不该写进已安装包的目录。
    calibration_file: Path | None = None


# 本进程当前配置。默认让库能独立跑；宿主启动时 configure() 换掉它。
_active: AnalyzerConfig = AnalyzerConfig()


def configure(cfg: AnalyzerConfig) -> None:
    """为本进程安装分析器配置。

    启动时、在 ``load_models()`` / ``analyze()`` 之前调用一次。之后分析读这里。
    换成不同 ``espeak_language`` 是安全的：会清掉按词缓存的音素，避免旧口音残留。
    """
    global _active
    if cfg.espeak_language != _active.espeak_language:
        # 延迟 import：speech.py 会 import 本模块，顶层互相 import 会成环。
        # configure() 能被调用时，包 __init__ 其实已经 import 过 speech 了。
        from . import speech
        speech._phonemize_word.cache_clear()
    _active = cfg


def get_config() -> AnalyzerConfig:
    """返回当前生效的分析器配置。"""
    return _active
