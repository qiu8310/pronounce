# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""两个打分引擎共用的、与引擎无关的结果类型。

``pronounce.score.acoustic`` 和 ``pronounce.score.phoneme`` 都返回同一个
:class:`PronunciationResult`，这样 GUI 只读一种结构，不用管调度器选了哪个引擎。
放在独立的小包里（不塞进任一引擎），避免引擎互相依赖。

GUI 硬性需要的四个字段：``score``、``word_errors``、``prosody``、``transcription``。
其余都是带默认值的引擎专有扩展：声学引擎填 ``acoustic_*``，音素引擎填
``per_phone_distance`` / ``phoneme_score`` / ``recall`` / ``good_anchor`` 等，
GUI 用不到的会安全忽略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PronunciationResult:
    """一次发音对比的结果（引擎无关）。

    @dataclass 会自动生成 ``__init__`` / ``__repr__`` 等，下面每个带类型注解的
    字段都会变成构造参数。没有默认值的字段必须先写（score 等四个必填项）。

    必填（每个引擎都会填）：score、word_errors、prosody、transcription。
    其余带默认值，引擎只填自己算得出的。
    ``prosody`` 由引擎留空，宿主（mimora/prosody.py 或 CLI）根据波形再填，
    这样语调图在两种引擎下长得一样。
    """

    score: float                                  # 0-100 总分
    word_errors: list[dict[str, Any]]             # 每个读错的词：expected / heard
    prosody: dict[str, list[float]]               # 宿主填写；引擎返回 {}
    transcription: str                            # 识别器听到的内容
    passed: bool = False                          # 声学：score>=阈值；音素：档位>=pass_bucket
    feedback: str = ""                            # 给人看的摘要
    # scored=False 表示完全没打分（"none" 引擎）。GUI 此时显示「未打分」，
    # 而不是把 score/passed 当作成绩。
    scored: bool = True

    # --- 音素引擎的 0-5 档 ---
    # bucket：把 0-100 分校准成 0-5 档。-1 表示该引擎不分级（声学引擎），
    # GUI 就退回显示原始 0-100。
    bucket: int = -1
    # user_percent：档位映射成用户可见百分比（区间中点），用来满足
    # 「good >= 90% / reference >= 95%」这类产品口径。GUI 实际展示 grade。
    user_percent: float = 0.0
    # grade：档位再加 +/- 深浅（"4-" / "4" / "4+"）。声学引擎留空。
    grade: str = ""
    # grade_value：把 grade 映到连续 0-5 轴（4-/4/4+ → 3.67/4.0/4.33），
    # 用来算会话平均和趋势箭头。未分级时为 -1.0。
    grade_value: float = -1.0

    # --- 两种引擎都会填、GUI 都读的展示字段 ---
    # field(default_factory=list) 而不是 = []：可变默认值若写成 = []，
    # 所有实例会共享同一个列表。factory 保证每个实例拿到新列表。
    words_with_errors: list[str] = field(default_factory=list)
    expected_phonemes: list[str] = field(default_factory=list)
    transcribed_phonemes: list[str] = field(default_factory=list)
    # word_diff：有词级错误时非空；空则 GUI 显示「与目标一致」。
    # 声学引擎：每个分歧片段一对 {expected, heard}；音素引擎：列出分歧词。
    word_diff: list[dict[str, str]] = field(default_factory=list)
    # reference_words：目标短语每个词 {word, correct}，按顺序，给「Phrase」行红绿高亮。
    reference_words: list[dict[str, Any]] = field(default_factory=list)
    # recognized_units：识别结果，每个 {unit, correct}。
    # 声学引擎的 unit 是词，音素引擎是音素；GUI 同一套渲染。
    recognized_units: list[dict[str, Any]] = field(default_factory=list)
    # weak_phonemes：参考音素里读得最差的几个，按严重程度降序，
    # 每项 {phoneme, severity, count}。声学引擎留空（没有逐音素分解）。
    weak_phonemes: list[dict[str, Any]] = field(default_factory=list)
    # ipa_words：按空白切词后的逐词音素对齐，给 IPA 诊断面板（音素引擎）。
    # 每项含 word、expected、heard、ok（平行列表）、start_sec、end_sec。
    ipa_words: list[dict[str, Any]] = field(default_factory=list)

    # --- 声学引擎诊断；别处可忽略 ---
    acoustic_distance: int = 0                     # 整段 Wav2Vec2 嵌入的 DTW 距离
    acoustic_per_step: float = 0.0                 # 每个对齐步的 DTW 距离（真正打分用的）
    acoustic_baseline: float = 0.0                 # 随机配对距离（本句的「完全不像」上限）

    # --- 音素引擎诊断；别处可忽略 ---
    per_phone_distance: float = 0.0               # 每个参考音素上的特征距离
    bad_baseline: float = 0.0                     # 本句「完全读错」锚点
    phoneme_score: float = 0.0                    # 0-100 发音质量分量
    recall: float = 0.0                           # 0-1，参考音素被覆盖的加权召回
    good_anchor: float = 0.0                      # 实际用到的 GOOD 锚（ceiling 模式下按句变）


# 声明本模块公开 API；from pronounce.common import * 时只导入这些名字。
__all__ = ["PronunciationResult"]
