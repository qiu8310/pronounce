# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""音素发音引擎的配置。

相对声学核：用短语**文本**（espeak 参考音素）和用户音频的音素 ASR 打分，
不需要每句参考录音。

同样与 GUI / 宿主无关：可调项都在 :class:`AnalyzerConfig`。库自带默认值，
import 后直接调 :func:`pronounce.score.phoneme.analyze` 即可。宿主启动时
:func:`configure` 注入一次，之后只读当前配置。

数据拟合出来的打分常数（GOOD 锚、召回阈值、轴权重、插入上限、档位）
不在这里：它们在引擎旁边的按语言模型校准文件
（``<lang>_model_calibration.json``，入库），本机 ``calibration.json``
（gitignored）只按用户覆盖 GOOD 锚——这样能和评估/校准工具共享，又不进源码。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalyzerConfig:
    """音素分析器读的设置。frozen=True 表示实例不可变，改配置请重新 configure。"""

    # 输出 espeak 风格 IPA 音素的 wav2vec2 CTC 模型。音素表和 espeak 参考对齐，
    # 没有清单不一致的问题。只在首次下载时比词 ASR 更重（约 1.2 GB）。
    model_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    # 识别器设备（"cuda"/"cpu"）。
    device: str = "cpu"
    # 参考文本转音素用的 espeak 口音（"en-us"/"en-gb"）。
    espeak_language: str = "en-us"
    # 0-100 分达到或超过此值视为合格。
    score_threshold: float = 70.0
    # 音素质量轴的 GOOD 锚模式：
    #   "global"  -- 模型校准里那一个 PHONEME_GOOD，可被 calibration.json 按用户覆盖；
    #                0-5 档切点是在这个锚下拟合的，生产分数和档位才对得上。
    #   "ceiling" -- 每句 GOOD = TTS 参考自己的逐步距离，完美跟读映射到 100
    #                （默认；需要宿主传入参考音频）。
    # 参考缺失/为空时静默退回 "global"（不会失败）。
    good_mode: str = "ceiling"
    # 引擎日志目录（和声学包对称）。
    log_dir: Path = Path("logs")
    # 当前练习用户；留给按人校准（未设则为 ""）。
    user_name: str = ""
    # 本机用户校准读写路径。None 时用包旁边的 calibration.json。
    # 宿主应注入自己的路径：那是机器本地状态，和入库的
    # <lang>_model_calibration.json 不同，也不该写进已安装包目录。
    calibration_file: Path | None = None


_active: AnalyzerConfig = AnalyzerConfig()


def configure(cfg: AnalyzerConfig) -> None:
    """为本进程安装分析器配置。在 ``load_models()`` / ``analyze()`` 之前调用一次。"""
    global _active
    _active = cfg


def get_config() -> AnalyzerConfig:
    """返回当前生效的分析器配置。"""
    return _active
