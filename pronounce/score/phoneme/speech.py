# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""音素发音引擎（仅凭文本即可打分，不必先录一句参考音频）。

流水线::

    文本 --espeak-ng--> 参考 IPA 音素 -+
                                        +-- 按发音特征加权的编辑距离
    用户音频 --wav2vec2 音素 ASR--> 音素 --+        |
                                                     v
                                          分数 (0-100) + 逐词 / 逐音素标记

与声学引擎不同：本引擎打分不需要每句的参考*录音*——参考音素由 espeak 从文本生成。
参考音频仍可传入；默认 ``good_mode="ceiling"`` 时会识别一次参考录音，把「完美朗读」
锚定到该句的 100 分；另一种 ``good_mode="global"`` 则对照校准得到的单一 PHONEME_GOOD
（与 0–5 分桶拟合时的方式一致）。

对外 API 与 ``acoustic/`` 完全对齐，调度器可以把两套引擎当同样的接口用：
    load_models()  -- 一次性加载 wav2vec2 音素权重（建议在线程里调用）。
    warm_up()      -- 空跑一遍，去掉首次调用的延迟。
    analyze(...)   -- 唯一入口，返回 PronunciationResult。

实现要点：
    * 识别器处理内存里的波形，不走文件路径，因此只缓存模型本身。
    * panphon 自带的数据是 UTF-8；在默认编码为 cp1252 的 Windows 进程里加载会
      抛 UnicodeDecodeError。我们不无条件 monkey-patch ``pathlib.Path.open``，而是
      先正常建表，仅在该错误时窄范围地默认 UTF-8（应用以 UTF-8 运行时永远不会
      走到这条路——启动时设 PYTHONUTF8=1）。
    * 打分常数（GOOD 锚点、召回阈值、轴权重、插入上限/门控、分桶）从本模块旁
      的按语言模型文件加载（``<lang>_model_calibration.json``，已提交），按配置的
      espeak 语言选取。机器本地的 ``calibration.json``（gitignore）只按语言和用户
      覆盖 ``phoneme_good``。``config.AnalyzerConfig`` 只放主机设置。

本模块不接触 GUI。
"""

# 推迟求值类型注解：写成 list[str]、str | None 时，运行时不会立刻去解析这些类型。
# 这样 3.9 风格的泛型注解可以写进源码，也避免「类还没定义完就被注解引用」的问题。
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# 与声学引擎共用的中性结果类型，GUI 无论当前哪套引擎都读同一份字段形状。
# 本引擎填写音素相关字段（per_phone_distance / phoneme_score / recall / good_anchor）；
# acoustic_* 字段保持默认值。
from pronounce.common import PronunciationResult

# 共用的波形预处理（pronounce.common 里一份、不依赖 torch）。
# as 起一个带下划线的别名，保留本模块一直以来的本地名，调用处不用改。
from pronounce.common.audio import (
    TARGET_SAMPLE_RATE,
    prepare_waveform as _prepare_waveform,
    waveform_digest,
)

# 捆绑的 espeak-ng 注册，与声学引擎共用，而不是每套引擎复制一份。
# 引擎自己注册，而不依赖 Kokoro/misaki 的 import 副作用；原因见该模块。
from pronounce.common.espeak import ensure_espeak

# 设置来自库自己的 AnalyzerConfig（见 config.py），绝不来自主机应用：
# 主机在启动时通过 configure() 注入一次自己的值。
from .config import get_config

# =====================================================================
# 常数与配置。
# =====================================================================
# wav2vec2 识别器要求严格 16 kHz 单声道：TARGET_SAMPLE_RATE
# （上面从 pronounce.common.audio 导入）。
# Kokoro 合成是 24 kHz；仅作为本引擎「单独使用」时参考音频的默认采样率。
# Mimora 的 main.py 总会显式传入 reference_sr（当前 TTS 后端的原生速率，
# 例如 Supertonic 西班牙语是 44.1 kHz），所以那里永远用不到这个默认值。
KOKORO_SAMPLE_RATE = 24_000

# 可调打分常数放在本模块旁边的 JSON 里，分成两层，避免「某个用户的校准」
# 污染引擎共享的基线：
#   * <lang>_model_calibration.json -- 模型层校准：锚点、分桶、门控、轴权重。
#     提交进仓库，按配置的 espeak 语言选取（"en-us" -> "en"）。
#   * calibration.json             -- 本机用户覆盖（gitignore）。
#     只按语言和用户存放重新锚定的 ``phoneme_good``，形状为
#     {lang: {"users": {user_name: {"phoneme_good": float, ...}}}}；
#     只覆盖模型的 ``phoneme_good``，其它键不动。
# 文件缺失或损坏时静默退回字面量默认值，打分不会因此中断。
# 以 ``_`` 开头的键（``_meta``）只是说明信息，不算打分常数。
# Path(__file__) 是本文件路径；.resolve() 变成绝对路径；.parent 是所在目录。
_DIR = Path(__file__).resolve().parent
_DEFAULT_LANG = "en"
# 无主机时用户校准文件的默认位置：就在本包旁边，独立调用方和评估工具都按这个找。
# 主机可通过 AnalyzerConfig.calibration_file 覆盖——真正读写都走
# current_calibration_file()，而不是直接用这个常量。
CALIBRATION_FILE = _DIR / "calibration.json"

# 固有打分常数——不属于「用数据拟合出来」的校准，所以与语言/用户无关、始终固定。
BAD_MIN_SPAN = 0.10                                 # 保证 bad 严格大于 good，窗口不会塌成 0
BAD_BASELINE_DEFAULT = 0.5                          # 参考或识别序列为空时的「全错」上限

def _read_json(path: Path) -> dict:
    """从 *path* 读出一个 JSON 对象；文件不存在或内容坏了则返回 ``{}``。

    ``with path.open(...) as fh`` 是上下文管理器：离开 with 块时文件会自动关闭。
    json.load 把文件内容解析成 Python 的 dict/list；若根节点不是 dict 也当作失败。
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

def current_calibration_file() -> Path:
    """当前生效的用户校准文件：主机指定的路径，否则用包内默认。

    功能：每次读写都现问一遍，而不是 import 时算死。
    返回：校准 JSON 的 Path。configure() 可能在本模块 import 之后才跑，
    必须反映「此刻」的配置。
    """
    return get_config().calibration_file or CALIBRATION_FILE

def _lang_key(espeak_language: str) -> str:
    """把 espeak 方言（"en-us"/"en-gb"）映射成校准语言键（"en"）。"""
    return (espeak_language or "").split("-")[0] or _DEFAULT_LANG

def _model_calibration_path(lang: str) -> Path:
    """已提交的模型校准文件路径（例如 en_model_calibration.json）。"""
    return _DIR / f"{lang}_model_calibration.json"

def _load_model_calibration(lang: str) -> tuple[dict, Path]:
    """读取 *lang* 的模型校准；没有该文件时回退到英语。

    返回值类型 ``tuple[dict, Path]`` 表示「一个二元组：字典 + 路径」。
    路径必须跟着数据一起返回，因为回退时数据是英语的、*lang* 却不是英语：
    若调用方自己拼文件名，日志会先写 "no model calibration for 'es'"，
    紧接着又写 "file=es_model_calibration.json"（那个文件其实不存在）。
    """
    path = _model_calibration_path(lang)
    data = _read_json(path)
    if not data and lang != _DEFAULT_LANG:
        logging.warning("[phoneme] no model calibration for %r; using %r defaults",
                        lang, _DEFAULT_LANG)
        path = _model_calibration_path(_DEFAULT_LANG)
        data = _read_json(path)
    return data, path

def _user_phoneme_good(lang: str, user_name: str) -> float | None:
    """该用户在 *lang* 下重新锚定的 ``phoneme_good``；未校准时为 None。

    ``float | None`` 表示「要么是浮点数，要么是 None」（联合类型）。
    用户文件形状：``{lang: {"users": {user_name: {"phoneme_good": float, ...}}}}``。
    .get() 在键不存在时返回 None，再配合 isinstance 逐层确认，避免坏 JSON 把程序打崩。
    """
    data = _read_json(current_calibration_file())
    lang_block = data.get(lang) if isinstance(data, dict) else None
    users = lang_block.get("users") if isinstance(lang_block, dict) else None
    entry = users.get(user_name) if isinstance(users, dict) else None
    if isinstance(entry, dict) and "phoneme_good" in entry:
        try:
            return float(entry["phoneme_good"])
        except (TypeError, ValueError):
            return None
    return None

# 数据拟合出来的打分常数。先用内置默认值声明，好让模块有一份明确的属性表
# （读者、IDE、静态分析都能看见）。下面的 _apply_model_calibration() 会在
# import 时、以及配置的语言/用户变化时，用模型校准 JSON 覆盖它们——
# 所以这些字面量只在校准文件缺某个键时生效（与那边 .get() 的回退值重复）。
# 每个常数的含义写在 _apply_model_calibration 内部。
PHONEME_GOOD = 0.0
BAD_SHRINK_PHONES = 12
BAD_CEILING = 0.40
INSERTION_CAP_PER_PHONE = 0.25
INSERTION_CONF_MIN = 0.0
INSERTION_CONF_AGG = "max"
RECALL_MAX_DIST = 0.13
WEIGHT_PHONEME = 0.7
WEIGHT_WORD = 0.3
WORD_GOOD_FRAC = 0.33
WORD_BAD_FRAC = 0.66
OVERPRODUCTION_TOLERANCE = 0.5
OVERPRODUCTION_STRENGTH = 1.0
BUCKET_CUTPOINTS: list[float] = []
BUCKET_TO_PERCENT: dict[str, Any] = {}
PASS_BUCKET = 4

def _apply_model_calibration(calib: dict) -> None:
    """把一份模型校准 dict 绑定到本模块的打分常数上。

    import 时先用英语默认值调用一次（库可单独使用，离线单测也有具体数值），
    主机的 espeak 语言已知后再由 ``_ensure_calibration`` 调一次。
    常数做成模块全局变量，测试仍可用 ``mock.patch.object(speech, ...)`` 去改。

    ``global`` 声明：下面赋值的是模块级变量，而不是函数内部的新局部变量。
    不写 global 的话，``PHONEME_GOOD = ...`` 只会在函数里新建一个同名局部名。
    """
    global PHONEME_GOOD, BAD_SHRINK_PHONES, BAD_CEILING
    global INSERTION_CAP_PER_PHONE, INSERTION_CONF_MIN, INSERTION_CONF_AGG
    global RECALL_MAX_DIST, WEIGHT_PHONEME, WEIGHT_WORD
    global WORD_GOOD_FRAC, WORD_BAD_FRAC
    global OVERPRODUCTION_TOLERANCE, OVERPRODUCTION_STRENGTH
    global BUCKET_CUTPOINTS, BUCKET_TO_PERCENT, PASS_BUCKET

    # 音素质量轴的锚点（见 _score_from_distance / _bad_baseline）。
    PHONEME_GOOD = calib.get("phoneme_good", 0.0)           # 逐音素距离被打成 100 分的那一端
    BAD_SHRINK_PHONES = calib.get("bad_shrink_phones", 12)  # 短句把 bad 往上抬的强度
    BAD_CEILING = calib.get("bad_ceiling", 0.40)            # 保守的「全错」锚点上限

    # 插入上限与置信度门控（防止识别器胡编音素把距离撑爆）。
    INSERTION_CAP_PER_PHONE = calib.get("insertion_cap_per_phone", 0.25)
    INSERTION_CONF_MIN = calib.get("insertion_conf_min", 0.0)    # tau；0 表示用 argmax 基线、不滤
    INSERTION_CONF_AGG = calib.get("insertion_conf_agg", "max")  # "max" 或 "mean"

    # 召回轴与最终混合（对应声学引擎的质量/词分拆）。
    RECALL_MAX_DIST = calib.get("recall_max_dist", 0.13)   # 小于此距离的音素算「召回了」
    WEIGHT_PHONEME = calib.get("weight_phoneme", 0.7)
    WEIGHT_WORD = calib.get("weight_word", 0.3)
    # 逐词三色高亮的分界，是 [good, bad] 窗口上的比例
    # （0 = 完美，1 = 完全错）：<= good_frac 为 "good"，>= bad_frac 为 "bad"，中间为 "ok"。
    WORD_GOOD_FRAC = calib.get("word_good_frac", 0.33)
    WORD_BAD_FRAC = calib.get("word_bad_frac", 0.66)

    # 过量产出惩罚：说出的音素远多于参考时，按 [0, 1] 缩放惩罚；落在容差带内为 0。
    # strength 为 0 则关闭。
    OVERPRODUCTION_TOLERANCE = calib.get("overproduction_tolerance", 0.5)
    OVERPRODUCTION_STRENGTH = calib.get("overproduction_strength", 1.0)

    # 0–5 分桶：把原始 0–100 分粗化成人校准过的档。块为空/缺失则关闭分桶
    # （bucket 保持 -1，用原始阈值）。
    buckets = calib.get("buckets", {})
    # 升序分数阈值；bucket = 分数跨过了几道门槛（0..len）。
    BUCKET_CUTPOINTS = [float(c) for c in buckets.get("cutpoints", [])]
    # bucket（转成 str）-> [lo, hi] 面向用户的百分区间（展示时取中点）。
    BUCKET_TO_PERCENT = calib.get("bucket_to_percent", {})
    # 达到或超过此桶算「通过」（好的朗读一般是 4–5 桶）。
    PASS_BUCKET = calib.get("pass_bucket", 4)

# import 时先套上英语默认值，模块可单独用（离线单测也有具体数字）。
# 主机用 configure() 注入配置后，_ensure_calibration() 再按语言重载并套上用户覆盖。
_apply_model_calibration(_load_model_calibration(_DEFAULT_LANG)[0])
# 上面那批常数当前对应的语言/用户；None 迫使第一次 _ensure_calibration() 必加载，
# 哪怕语言正好是默认英语。``str | None`` = 字符串或尚未加载。
_loaded_lang: str | None = None
_loaded_user: str | None = None
# threading.Lock：互斥锁。校准重载会改写很多模块全局变量，两次并发的首次调用
# 会把写入交错。锁只管「重载本身」——并不保证 analyze() 在重载中途读全局是安全的。
# 应用里 GUI 会串行化 analyze()；库用户一边分析一边改语言仍会竞态（已知限制）。
_calibration_lock = threading.Lock()

def _ensure_calibration() -> None:
    """若配置的语言/用户变了，则加载对应的模型校准 + 用户校准。

    对应声学引擎的 ``current_acoustic_floor``：模型分桶和锚点跟 ``espeak_language``，
    用户重新锚定的 ``phoneme_good``（由 calibrate.py 写入）覆盖模型默认。
    只在语言或用户真的变了才重载，稳态分析零额外开销。
    ``with _calibration_lock``：进入加锁，离开（含异常）时自动解锁。
    """
    with _calibration_lock:
        _ensure_calibration_locked()

def _ensure_calibration_locked() -> None:
    """假定调用方已持有 _calibration_lock，真正执行校准重载。"""
    global _loaded_lang, _loaded_user, PHONEME_GOOD
    cfg = get_config()
    lang = _lang_key(cfg.espeak_language)
    user = cfg.user_name
    if lang == _loaded_lang and user == _loaded_user:
        return
    model, model_file = _load_model_calibration(lang)
    _apply_model_calibration(model)
    user_good = _user_phoneme_good(lang, user)
    if user_good is not None:
        PHONEME_GOOD = user_good
    # 把实际加载内容打成 JSON 记入日志，有效打分配置可从 logs/main.log 还原。
    # 模型层与用户覆盖分行。只在（重新）加载时记——即启动一次、语言/用户变化再记——
    # 不会淹没每条 take 的分析日志。
    # `file` 是真正读到的文件；英语回退时它不是按 `lang` 命名的那个——见 _load_model_calibration。
    logging.info(
        "[phoneme] model calibration loaded (lang=%s, file=%s): %s",
        lang, model_file.name,
        json.dumps(model, ensure_ascii=False),
    )
    logging.info(
        "[phoneme] user calibration loaded (lang=%s, user=%r, file=%s): %s",
        lang, user, current_calibration_file().name,
        json.dumps({
            "user_phoneme_good": user_good,
            "effective_phoneme_good": PHONEME_GOOD,
        }, ensure_ascii=False),
    )
    _loaded_lang, _loaded_user = lang, user

# =====================================================================
# 样本日志 + 校准写回（对应声学引擎）。
# analyze() 每次 take 往 logs/phoneme_samples.jsonl 追加一条；离线
# phoneme/calibrate.py 用其中的参考自测重新锚定 PHONEME_GOOD。
# 与声学的 acoustic_samples.jsonl 分开，两套引擎的日志互不混。
# =====================================================================
def samples_file() -> Path:
    """音素样本日志路径（在主机配置的 log 目录下）。

    功能：拼出 phoneme_samples.jsonl 的完整路径。
    返回：Path，目录来自 AnalyzerConfig.log_dir。
    """
    return Path(get_config().log_dir) / "phoneme_samples.jsonl"

# 样本日志上限：每次进程里、第一次追加之前，把日志裁到最新 MAX_SAMPLES_KEPT 行，
# 避免无限变长。明显高于 calibrate.py 的 MAX_SAMPLES_USED（300），裁剪不会饿死校准。
MAX_SAMPLES_KEPT = 2000

_samples_trimmed = False  # 进程内只裁一次，给 _trim_sample_log() 当开关

def _trim_sample_log() -> None:
    """丢掉样本日志里除最新 MAX_SAMPLES_KEPT 行以外的内容。

    由 _append_sample 每个进程调用一次，稳态追加仍是 O(1) 写，文件只在两次运行之间变短。
    """
    path = samples_file()
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= MAX_SAMPLES_KEPT:
        return
    path.write_text("\n".join(lines[-MAX_SAMPLES_KEPT:]) + "\n", encoding="utf-8")
    logging.info("[phoneme] sample log trimmed: %d -> %d lines (%s)",
                 len(lines), MAX_SAMPLES_KEPT, path.name)

def _append_sample(record: dict[str, Any]) -> None:
    """往样本日志追加一条分析记录（尽力而为）。

    写失败绝不能打断分析本身，所以所有错误只记日志再吞掉
    （对应声学引擎的 _append_calibration_sample）。
    ``dict[str, Any]`` = 键是字符串、值是任意类型的字典。
    """
    global _samples_trimmed
    try:
        if not _samples_trimmed:
            _samples_trimmed = True  # 先置位：裁剪失败也不能每条 take 重试
            _trim_sample_log()
        path = samples_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logging.exception("[phoneme] failed to append sample")

def current_lang() -> str:
    """当前配置的 espeak 语言对应的校准语言键（"en"、"es"）。

    功能：把配置里的方言压成校准 JSON 用的短键。
    返回：如 ``en``、``es``。
    """
    return _lang_key(get_config().espeak_language)

def current_phoneme_good() -> float:
    """当前语言/用户生效的 GOOD 锚点。

    功能：确保校准已加载后读出 PHONEME_GOOD。
    返回：逐音素距离被映射成 100 分的那一端。
    """
    _ensure_calibration()
    return PHONEME_GOOD

def save_calibration(phoneme_good: float, extra: dict[str, Any] | None = None) -> None:
    """把当前用户重新锚定的 ``phoneme_good`` 写入用户校准文件。

    功能：供 phoneme/calibrate.py 调用。写入
    ``{lang: {"users": {user_name: {...}}}}``，文件是本机 calibration.json（gitignore），
    按配置的 espeak 语言和用户名分键；其它用户、其它语言的条目保留。
    已提交的按语言模型校准（``<lang>_model_calibration.json``：分桶、门控、默认锚点）
    永不改动。新值对本进程立即生效，下次启动也生效。
    参数：
        phoneme_good: 新的 GOOD 锚点。
        extra: 可选，合并进该用户条目的额外字段；``dict | None`` 表示「字典或省略」。
    返回：无。
    """
    global PHONEME_GOOD, _loaded_lang, _loaded_user
    cfg = get_config()
    lang = _lang_key(cfg.espeak_language)
    user = cfg.user_name
    # 这里只解析一次路径：本函数先读后写，中间若配置被换掉，读写必须仍是同一文件。
    path = current_calibration_file()

    data = _read_json(path)
    lang_block = data.get(lang)
    if not isinstance(lang_block, dict):
        lang_block = {}
    users = lang_block.get("users")
    if not isinstance(users, dict):
        users = {}

    entry: dict[str, Any] = {
        "phoneme_good": round(float(phoneme_good), 6),
        "user_name": user,
        "recalibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        entry.update(extra)
    users[user] = entry
    lang_block["users"] = users
    data[lang] = lang_block

    # 主机的目录在 config.py import 时就会建好；独立调用方可能把路径指到新地方。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 立刻应用到本进程（与声学引擎一致）。
    PHONEME_GOOD = float(phoneme_good)
    _loaded_lang, _loaded_user = lang, user
    logging.info("[phoneme] wrote phoneme_good=%.6f (lang=%s user=%r) -> %s",
                 phoneme_good, lang, user, path)

# =====================================================================
# 模型生命周期。
# =====================================================================
_processor = None
_model = None
_load_lock = threading.Lock()

def _ensure_phonemizer_detected() -> None:
    """让 transformers 把 phonemizer-fork 当成它要的 "phonemizer" 后端。

    我们装的是 ``phonemizer-fork``（不是原版 ``phonemizer``），因为 Kokoro/misaki
    需要它的 ``EspeakWrapper.set_data_path()``；两者共用 ``phonemizer`` 这个 import
    名，不能同时装。transformers>=5 能检测（按 import spec 查），但 transformers<5
    —— Intel macOS 会落到这个版本，因为那边没有 torch>2.2 的 x86_64 轮子——
    查的是 *发行包* 名，看到的是 ``phonemizer-fork`` 而不是 ``phonemizer``，于是
    ``Wav2Vec2PhonemeCTCTokenizer`` 拒绝初始化。fork 是能用的替代品，所以能 import
    时就把 transformers 缓存的可用性标志拨成 True。
    phonemizer 真的没装时什么都不做（让 transformers 自己报错）；已经检测到时
    （transformers>=5）也是空操作。

    ``import importlib.util`` 写在函数里：延迟导入，只有走到这里才加载，加快模块 import。
    """
    import importlib.util

    if importlib.util.find_spec("phonemizer") is None:
        return
    try:
        from transformers.utils import import_utils as _iu
    except Exception:
        return
    try:
        if _iu.is_phonemizer_available():
            return
    except Exception:
        return
    # transformers<5 的 is_phonemizer_available() 读的就是这个模块级标志。
    if hasattr(_iu, "_phonemizer_available"):
        _iu._phonemizer_available = True

def load_models() -> None:
    """把 wav2vec2 音素权重加载进内存，只做一次。重复调用是安全的。

    功能：在锁保护下加载 AutoProcessor / CTC 模型。很重（首次约 1.2 GB 下载）；
    应在模式启动时从后台守护线程调用，以免卡住 GUI（对应 acoustic.load_models）。
    返回：无。
    ``from transformers import ...`` 在函数内部：延迟导入，没加载模型时不拖进 torch。
    """
    global _processor, _model
    with _load_lock:
        if _model is not None and _processor is not None:
            return
        from transformers import AutoModelForCTC, AutoProcessor  # pyright: ignore[reportAttributeAccessIssue]

        from pronounce.common.compat import allow_torch_load_for_trusted_models

        _ensure_phonemizer_detected()
        # 必须在 processor 加载之前注册捆绑的 espeak-ng：wav2vec2-phoneme 的
        # tokenizer 会在 from_pretrained 里建一个 phonemizer EspeakBackend，
        # 否则会报 "espeak not installed"。
        ensure_espeak()
        allow_torch_load_for_trusted_models()
        cfg = get_config()
        _processor = AutoProcessor.from_pretrained(cfg.model_name)
        _model = AutoModelForCTC.from_pretrained(cfg.model_name).to(cfg.device).eval()

def warm_up() -> None:
    """空跑几遍，去掉首次调用延迟（识别器 + panphon + espeak）。

    功能：加载模型、校准，再对静音和短文本各走一遍热身路径。
    返回：无。某一步失败只记日志，不向外抛。
    """
    load_models()
    _ensure_calibration()
    cfg = get_config()
    dummy = np.zeros(TARGET_SAMPLE_RATE // 2, dtype=np.float32)  # 0.5 秒静音
    try:
        _spoken_from_wav(dummy, cfg.device)
    except Exception:
        logging.exception("[phoneme] recognizer warm-up failed")
    try:
        _feature_table()                                   # 建一次 panphon 特征表
        reference_phonemes("warm up", cfg.espeak_language)  # 拉起一次 espeak
    except Exception:
        logging.exception("[phoneme] scoring warm-up failed")

def _ensure_loaded() -> None:
    """推理前保证识别器已经在内存里。"""
    if _model is None or _processor is None:
        load_models()

# espeak 注册在 pronounce.common.espeak（上面已导入 ensure_espeak），
# 与声学引擎共用，修一处两边都生效。不要在这里再私藏一份副本，否则另一套引擎未注册。

# 音频预处理在 pronounce.common.audio（上面已导入 _prepare_waveform），
# 与声学引擎、主机的韵律层共用，大家量的是同一份预处理后的信号。

# =====================================================================
# 第 2 步 —— 用 wav2vec2 音素识别器从音频得到「说出的音素」。
# 处理内存中的 16 kHz 波形（没有文件路径），因此不会因临时文件泄漏路径缓存；
# 只缓存模型本身（见 load_models）。
# =====================================================================
def _recognize_argmax(wav16: np.ndarray, device: str) -> str:
    """贪心 CTC 解码 -> 空格分隔的 espeak/IPA 音素。

    CTC：按帧输出音素 id，相邻相同的要合并、空白符要丢掉。
    ``with torch.no_grad()``：上下文管理器，告诉 PyTorch 这次不算梯度，省内存。
    """
    import torch

    _ensure_loaded()
    inputs = _processor(wav16, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = _model(inputs.input_values.to(device)).logits
    predicted_ids = logits.argmax(dim=-1)
    return _processor.batch_decode(predicted_ids)[0]

def _aggregate_conf(frame_confs: list[float]) -> float:
    """把一个 CTC 记号跨若干帧的后验概率汇成一个置信度（"max" 或 "mean"）。"""
    if not frame_confs:
        return 0.0
    if INSERTION_CONF_AGG == "mean":
        return sum(frame_confs) / len(frame_confs)
    return max(frame_confs)

def _ctc_phone_runs(wav16: np.ndarray, device: str
                    ) -> tuple[list[tuple[str, float, int, int]], int]:
    """贪心 CTC 折叠 -> (runs, n_frames)。

    每个 run 是 ``(token, conf, start_frame, end_frame_inclusive)``。
    ``n_frames`` 是整条 CTC 时间轴长度（含空白）。
    返回类型层层嵌套：二元组里第一项是「四元组列表」，第二项是帧数。
    getattr(..., None)：属性可能不存在时给默认值，避免 AttributeError。
    """
    import torch

    _ensure_loaded()
    inputs = _processor(wav16, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = _model(inputs.input_values.to(device)).logits
    probs = logits.softmax(dim=-1)[0]                 # (T, vocab)
    frame_ids = probs.argmax(dim=-1).tolist()         # (T,) 贪心 CTC 路径
    frame_conf = probs.max(dim=-1).values.tolist()    # (T,) 选中 id 的后验

    tokenizer = _processor.tokenizer
    blank_id = tokenizer.pad_token_id                 # wav2vec2 里 CTC blank == pad
    delimiter = getattr(tokenizer, "word_delimiter_token", None)

    runs: list[tuple[str, float, int, int]] = []
    n = len(frame_ids)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and frame_ids[j + 1] == frame_ids[i]:
            j += 1
        token_id = frame_ids[i]
        if token_id != blank_id:
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token and token != delimiter:
                runs.append((token, _aggregate_conf(frame_conf[i : j + 1]), i, j))
        i = j + 1
    return runs, n

def _recognize_with_conf(wav16: np.ndarray, device: str) -> list[tuple[str, float]]:
    """贪心 CTC 解码，同时返回每个音素的后验置信度。"""
    runs, _n = _ctc_phone_runs(wav16, device)
    return [(tok, conf) for tok, conf, _i, _j in runs]

def _spoken_timed_from_wav(wav16: np.ndarray, device: str) -> list[dict[str, Any]]:
    """识别出的音素，带上在 ``wav16`` 上的近似起止秒。

    音素序列与打分一致（同一套置信度门控 + 音素清单折叠）。
    时间把 CTC 帧 run 按比例映射到波形时长，供逐词切片播放。
    """
    runs, n_frames = _ctc_phone_runs(wav16, device)
    if INSERTION_CONF_MIN > 0:
        runs = [r for r in runs if r[1] >= INSERTION_CONF_MIN]
    if n_frames <= 0:
        return []
    duration = float(len(wav16)) / float(TARGET_SAMPLE_RATE)
    sec_per_frame = duration / n_frames
    raw_spans = [
        {"phone": tok, "t0": i * sec_per_frame, "t1": (j + 1) * sec_per_frame}
        for tok, _conf, i, j in runs
    ]
    return _normalize_phone_spans(raw_spans)

def _spoken_from_wav(wav16: np.ndarray, device: str) -> list[str]:
    """识别说话人实际发出的音素，并做归一化与清单折叠。"""
    return [span["phone"] for span in _spoken_timed_from_wav(wav16, device)]

# =====================================================================
# 第 1 步 —— 从文本得到参考音素（phonemizer 调 espeak-ng）。
# =====================================================================
def reference_phonemes(text: str, espeak_lang: str) -> list[str]:
    """把 ``text`` 转成扁平的 IPA 音素符号列表（逐音素分开）。

    功能：按词分组后再拍平，得到整句音素序列。
    参数：
        text: 要朗读的参考句子。
        espeak_lang: espeak 语言/方言，如 ``en-us``。
    返回：音素字符串列表，例如 ``["h", "ə", "l", "oʊ"]``。
    """
    return [p for word in reference_word_phonemes(text, espeak_lang) for p in word]

def reference_word_phonemes(text: str, espeak_lang: str) -> list[list[str]]:
    """把 ``text`` 按空白切成词，每个词一组音素，顺序与词一致。

    功能：返回的组数与 ``text.split()`` 的 token 数严格 1:1
    （纯标点等产不出音素的 token 对应空列表），方便打分器把音素错误映射回 GUI 高亮的那个词。
    参数：
        text: 参考句子。
        espeak_lang: espeak 语言代码。
    返回：``list[list[str]]``，外层一词一组，内层是该词的音素。

    以前是整句音素化再按 espeak 自己的词边界切，只要 espeak 拆词或丢 token
    （数字变成多个词、纯标点 token），就会和 ``text.split()`` 错位，后面所有词都涂错色。
    现在每个 token 单独音素化（一次批量 espeak 调用），边界与 GUI 一致。
    用 ``[p for w in groups for p in w]`` 拍平即可还原 ``reference_phonemes`` 的序列。
    """
    tokens = text.split()
    if not tokens:
        return []

    from phonemizer import phonemize
    from phonemizer.separator import Separator

    ensure_espeak()
    # 传入 list 时，一次后端调用里每个 token 独立音素化；
    # word="" 因为每项已经是一个 token，不需要再标词内边界。
    ipa_list = phonemize(
        tokens,
        language=espeak_lang,
        backend="espeak",
        strip=True,
        with_stress=False,
        preserve_punctuation=False,
        separator=Separator(phone=" ", word="", syllable=""),
    )
    # 部分 phonemizer 版本对长度为 1 的 list 返回裸 str；统一成 list。
    if isinstance(ipa_list, str):
        ipa_list = [ipa_list]
    return [_normalize_phones(_tokenize_ipa(ipa)) for ipa in ipa_list]

# =====================================================================
# 音素切分、去掉超音段变音符号、音素清单折叠。
# =====================================================================
def _tokenize_ipa(ipa: str) -> list[str]:
    """把 IPA 字符串切成音素符号（有空白就按空白切，否则按字符切）。"""
    ipa = ipa.strip()
    if " " in ipa or "\n" in ipa:
        return [tok for tok in ipa.split() if tok]
    return [ch for ch in ipa if not ch.isspace()]

# 一边标了、另一边没标的超音段变音符号（espeak vs w2v2 识别器）。
# 写成码点，避免源码里出现裸的组合用变音符号。
_DIACRITIC_CODEPOINTS = (
    0x02D0, 0x02D1, 0x02B0, 0x02C0, 0x02C8, 0x02CC,
    0x0329, 0x030D, 0x032F, 0x0361, 0x035C,
)
# dict.fromkeys(码点) 得到「这些码点 -> None」的映射表，给 str.translate 当删除表。
_STRIP_DIACRITICS = dict.fromkeys(_DIACRITIC_CODEPOINTS)

# 音素清单折叠：espeak 参考（en-us）和识别器对同一声音用不同 IPA 习惯；
# 两边都规范化到同一套符号才能对齐。对西班牙语安全（西语 espeak 已经发这些
# 基本符号，表几乎是恒等映射）。
_PHONE_FOLD = {
    "ɹ": "r", "ɾ": "r", "ɻ": "r",     # 卷舌近音 / 闪音 / 卷舌 -> r
    "æ": "a", "ɐ": "a",               # 近开前元音 / 近开央元音 -> a
    "ᵻ": "ɪ", "ɨ": "ɪ",              # 弱化 / 央高元音 -> ɪ
    "ɚ": "ə", "ɝ": "ə",              # 带 r 色彩的 schwa -> 普通 schwa
    "oʊ": "o", "əʊ": "o",            # 美音 / 英音 "goat" 双元音 -> o
}

def _normalize_phones(tokens: list[str]) -> list[str]:
    """去掉超音段变音符号，并折叠音素清单，让参考侧与识别侧对齐。

    圆括号 ``(... for ...)`` 是生成器表达式：惰性逐个产生，不像方括号列表推导会一次建完。
    """
    cleaned = (tok.translate(_STRIP_DIACRITICS) for tok in tokens)
    folded = (_PHONE_FOLD.get(tok, tok) for tok in cleaned)
    return [tok for tok in folded if tok]

def _normalize_phone_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """与 ``_normalize_phones`` 同一套清单折叠，t0/t1 时刻跟着走、不错位。"""
    out: list[dict[str, Any]] = []
    for span in spans:
        tok = str(span.get("phone", "")).translate(_STRIP_DIACRITICS)
        tok = _PHONE_FOLD.get(tok, tok)
        if tok:
            out.append({"phone": tok, "t0": float(span["t0"]), "t1": float(span["t1"])})
    return out

# =====================================================================
# 发音特征距离（panphon）。延迟导入并缓存：没装 panphon 时模块仍能 import，
# 特征表只建一次。
# =====================================================================
# @lru_cache(maxsize=1)：把返回值记住；maxsize=1 表示只留最近 1 次，表永远只建一次。
@lru_cache(maxsize=1)
def _feature_table():
    """建一次 panphon 特征表（在 cp1252 的 Windows 上也按 UTF-8 读）。

    ``import panphon`` 放在函数里：延迟导入，import 本模块时不强制依赖 panphon。
    """
    import panphon

    try:
        return panphon.FeatureTable()
    except UnicodeDecodeError:
        # panphon 用 pathlib.Path.open 打开自带的 UTF-8 数据时没指定 encoding，
        # 默认编码是 cp1252 就会失败。只在这次失败时临时把默认改成 UTF-8，
        # 用完立刻恢复。应用以 UTF-8 运行（启动时 PYTHONUTF8=1）则永远不进这个分支。
        import pathlib

        original_open = pathlib.Path.open

        # 嵌套函数：临时顶替 Path.open，只在这次建表期间把缺省编码改成 UTF-8。
        def _utf8_open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
            if "b" not in mode and encoding is None:
                encoding = "utf-8"
            return original_open(self, mode, buffering, encoding, errors, newline)

        pathlib.Path.open = _utf8_open
        try:
            return panphon.FeatureTable()
        finally:
            # finally：无论 try 成功还是再抛错，都会执行，保证补丁被拆掉。
            pathlib.Path.open = original_open

@lru_cache(maxsize=4096)
def _phone_vector(phone: str):
    """一个音素的数值发音特征向量；未知音素则为 None。"""
    vectors = _feature_table().word_to_vector_list(phone, numeric=True)
    return tuple(vectors[0]) if vectors else None

@lru_cache(maxsize=8192)
def _substitution_cost(a: str, b: str) -> float:
    """两个音素的特征距离，落在 [0, 1]（0 = 相同；1 = 未知或不匹配）。"""
    if a == b:
        return 0.0
    va, vb = _phone_vector(a), _phone_vector(b)
    if va is None or vb is None or len(va) != len(vb):
        return 1.0
    differing = sum(1 for x, y in zip(va, vb) if x != y)
    return differing / len(va)

# =====================================================================
# 第 3 步 —— 按发音特征加权的编辑距离对齐与打分。
# =====================================================================
_OP_SUB, _OP_DEL, _OP_INS = "sub", "del", "ins"

def _edit_alignment(reference: list[str],
                    spoken: list[str]) -> tuple[list[tuple[str, str]], float]:
    """在音素 token 上做按特征加权的编辑距离对齐。

    返回对齐后的 ``(参考, 识别)`` 对（``""`` 表示插入或删除）以及总分式距离。
    手写 DP 是因为 ``Levenshtein`` 按字符算，不能处理「多字符 IPA token 的列表」，
    而且我们需要每个音素自己的特征替换代价。
    ``list[list[str | None]]``：二维表，格子里是操作名字符串或尚无指针时的 None。
    """
    n, m = len(reference), len(spoken)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
        back[i][0] = _OP_DEL
    for j in range(1, m + 1):
        cost[0][j] = float(j)
        back[0][j] = _OP_INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitute = cost[i - 1][j - 1] + _substitution_cost(reference[i - 1], spoken[j - 1])
            delete = cost[i - 1][j] + 1.0
            insert = cost[i][j - 1] + 1.0
            best = min(substitute, delete, insert)
            cost[i][j] = best
            back[i][j] = _OP_SUB if best == substitute else (_OP_DEL if best == delete else _OP_INS)

    pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == _OP_SUB:
            pairs.append((reference[i - 1], spoken[j - 1]))
            i, j = i - 1, j - 1
        elif op == _OP_DEL:
            pairs.append((reference[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", spoken[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs, cost[n][m]

def _capped_per_phone_distance(pairs: list[tuple[str, str]],
                               distance: float, n_reference: int) -> float:
    """逐音素距离，但插入代价按「每个参考音素」封顶。

    插入随*识别*长度增长，胡编的识别器可以把距离撑到无限大，正确朗读也会被打到地板。
    插入部分上限为 ``INSERTION_CAP_PER_PHONE * n_reference``，替换/删除（真错误）不动。
    """
    if INSERTION_CAP_PER_PHONE <= 0 or n_reference <= 0:
        return distance / n_reference if n_reference else 0.0
    insertion_cost = float(sum(1 for ref_sym, _ in pairs if not ref_sym))
    cap = INSERTION_CAP_PER_PHONE * n_reference
    capped = (distance - insertion_cost) + min(insertion_cost, cap)
    return capped / n_reference

def _bad_baseline(reference: list[str], spoken: list[str]) -> float:
    """本句的「完全说错」锚点：所有音素对的平均特征距离。"""
    if not reference or not spoken:
        return BAD_BASELINE_DEFAULT
    total = sum(_substitution_cost(r, s) for r in reference for s in spoken)
    observed = total / (len(reference) * len(spoken))
    return _widen_bad_for_length(observed, len(reference))

def _widen_bad_for_length(observed_bad: float, n_reference: int) -> float:
    """把短句的 ``bad`` 锚点往保守上限抬一抬。

    短参考的 ``bad`` 噪声大、常常偏低，口音正确的朗读会被压到地板。
    用随长度变化的信任权重向 ``BAD_CEILING`` 混合；``max(observed, ceiling)``
    保证只*加宽*窗口（真垃圾句仍打地板）。BAD_SHRINK_PHONES == 0 则关闭。
    """
    if BAD_SHRINK_PHONES <= 0:
        return observed_bad
    trust = n_reference / (n_reference + BAD_SHRINK_PHONES)
    return trust * observed_bad + (1.0 - trust) * max(observed_bad, BAD_CEILING)

def _score_from_distance(per_phone_distance: float, bad: float, good: float) -> float:
    """把逐音素距离对照 [good, bad] 窗口映射到 0–100。"""
    span = max(bad - good, BAD_MIN_SPAN)
    accuracy = 1.0 - (per_phone_distance - good) / span
    return round(max(0.0, min(1.0, accuracy)) * 100.0, 1)

def _weak_phonemes(pairs: list[tuple[str, str]],
                   max_count: int = 3) -> list[dict[str, Any]]:
    """参考音素里读得最差的几个，给 GUI「重点练这些」那一行。

    对齐里每个参考音素（替换或删除）算一个 [0, 1] 严重度：听到的音素的特征距离，
    完全没发出则为 1.0。小于 ``RECALL_MAX_DIST`` 的算正确（与「听到」行同一阈值）并跳过。
    其余按符号汇总——严重度相加，又多又差的音素会排到前面——返回最差的 ``max_count`` 个，
    从重到轻，每项 ``{"phoneme", "severity", "count"}``。

    插入（``ref_sym == ""``）没有可归咎的参考音素，忽略。干净朗读返回空列表。
    ``key=lambda kv: ...``：排序时用这个无名小函数从 (符号, 统计) 里取出严重度。
    """
    totals: dict[str, dict[str, Any]] = {}
    for ref_sym, hyp_sym in pairs:
        if not ref_sym:                       # 插入：无从归属
            continue
        severity = 1.0 if not hyp_sym else _substitution_cost(ref_sym, hyp_sym)
        if severity < RECALL_MAX_DIST:        # 足够近 -> 算正确
            continue
        entry = totals.setdefault(ref_sym, {"severity": 0.0, "count": 0})
        entry["severity"] += severity
        entry["count"] += 1
    ranked = sorted(totals.items(), key=lambda kv: kv[1]["severity"], reverse=True)
    return [
        {"phoneme": sym, "severity": round(v["severity"], 4), "count": v["count"]}
        for sym, v in ranked[:max_count]
    ]

def _phoneme_recall(pairs: list[tuple[str, str]]) -> float:
    """参考音素里实际发出来的比例，从对齐结果直接读。"""
    recalled = 0
    total = 0
    for ref_sym, hyp_sym in pairs:
        if not ref_sym:                       # 插入：这里没有参考音素
            continue
        total += 1
        if hyp_sym and _substitution_cost(ref_sym, hyp_sym) < RECALL_MAX_DIST:
            recalled += 1
    return recalled / total if total else 0.0

# @dataclass：按下面列出的字段自动生成 __init__ / __repr__ 等，不必手写构造函数。
@dataclass
class ScoreResult:
    """双轴分数，加上对齐结果和原始逐音素距离。

    功能：``align_and_score`` 的返回值容器。
    字段（构造时当作参数传入）：
        score: 综合分 0–100（含过量产出惩罚）。
        pairs: 对齐后的 (参考, 识别) 对。
        per_phone_distance: 封顶后的逐音素距离。
        bad_baseline: 本句「全错」锚点。
        phoneme_score: 音素质量轴分数。
        recall: 音素召回率 0–1。
        good: 本句使用的 GOOD 锚点。
    """

    score: float
    pairs: list[tuple[str, str]]
    per_phone_distance: float
    bad_baseline: float
    phoneme_score: float
    recall: float
    good: float

def align_and_score(reference: list[str], spoken: list[str],
                    good: float | None = None) -> ScoreResult:
    """把 ``spoken`` 对齐到 ``reference``，按质量 + 召回两轴打分。

    功能：编辑距离对齐、封顶逐音素距离、映射到 0–100，再叠过量产出惩罚。
    参数：
        reference: 参考音素序列。
        spoken: 识别出的音素序列。
        good: 覆盖音素轴的 GOOD 锚点；``None`` 用全局 ``PHONEME_GOOD``；
            传入数值（ceiling 模式）则是参考录音自己的逐音素距离，该句完美朗读映射到 100。
    返回：ScoreResult。
    """
    if not reference:
        raise ValueError("empty reference phoneme sequence")

    pairs, distance = _edit_alignment(reference, spoken)
    per_phone_distance = _capped_per_phone_distance(pairs, distance, len(reference))
    bad = _bad_baseline(reference, spoken)
    good_anchor = PHONEME_GOOD if good is None else good
    phoneme_score = _score_from_distance(per_phone_distance, bad, good_anchor)
    recall = _phoneme_recall(pairs)
    base_score = WEIGHT_PHONEME * phoneme_score + WEIGHT_WORD * recall * 100.0
    # 惩罚严重过量产出：多说/说别的本来不花钱，会把分数和逐词颜色用碰巧匹配撑满（BUG-UI-2）。
    penalty = _overproduction_penalty(len(reference), len(spoken))
    score = round(base_score * (1.0 - penalty), 1)
    return ScoreResult(
        score=score,
        pairs=pairs,
        per_phone_distance=per_phone_distance,
        bad_baseline=bad,
        phoneme_score=phoneme_score,
        recall=recall,
        good=good_anchor,
    )

# =====================================================================
# 参考录音识别缓存（ceiling 模式的 GOOD 锚点 + 重试复用）。
# 同一句参考 take 每次重试都会打分；识别一次即可，对应声学引擎的 _reference_features。
# 键是内容哈希 + 采样率；缓存很小且会自清空，长会话也不会涨。锁对应声学的
# _reference_cache_lock：应用里 analyze() 已被 GUI 的 is_processing_audio 串行化，
# 但本包也作为独立库文档化，并发调用是合法的。
# dict 键类型 ``tuple[bytes, tuple[int, ...], int]``：哈希、波形 shape、采样率。
# ``tuple[int, ...]`` 里的 ``...`` 表示「任意多个 int」。
# =====================================================================
_ref_cache: dict[tuple[bytes, tuple[int, ...], int], list[str]] = {}
_ref_cache_lock = threading.Lock()

def _recognize_reference(reference_audio: np.ndarray, reference_sr: int,
                         device: str) -> list[str]:
    """参考 take 识别出的音素，按内容 + 采样率缓存。"""
    arr = np.asarray(reference_audio, dtype=np.float32)
    key = (waveform_digest(arr), arr.shape, reference_sr)
    with _ref_cache_lock:
        cached = _ref_cache.get(key)
    if cached is not None:
        return cached
    # 识别在锁外跑（这是最贵的一步）；并发重复最多再算一遍同样的值。
    spoken = _spoken_from_wav(_prepare_waveform(reference_audio, reference_sr), device)
    with _ref_cache_lock:
        if len(_ref_cache) >= 8:
            _ref_cache.clear()
        _ref_cache[key] = spoken
    return spoken

# =====================================================================
# 音素错误 -> 词映射（给 GUI 逐词高亮）。
# =====================================================================
def _word_recall(groups: list[list[str]],
                 pairs: list[tuple[str, str]]
                 ) -> tuple[list[int], list[list[str]], list[float]]:
    """每个参考词：召回的音素数、听到的音素、平均距离。

    ``groups`` 是按词分好的音素；``pairs`` 是拍平后参考序列上的对齐（同一顺序），
    所以可以走一张「音素下标 -> 词下标」表，把非插入对归到对应的词。

    ``word_dist[wi]`` 是该词参考音素的平均发音特征距离：替换贡献特征代价，
    *删除*（没发出）贡献最大值 1.0。与总分同一轴，逐词高亮才和分桶一致——
    「脏」或对调的词离参考远，就会显示为远，而不再是以前那种宽松的 50% 召回旗标。
    """
    word_of: list[int] = []
    for wi, group in enumerate(groups):
        word_of.extend([wi] * len(group))
    n_ref = len(word_of)

    recalled = [0] * len(groups)
    heard: list[list[str]] = [[] for _ in groups]
    dist_sum = [0.0] * len(groups)            # 每个词累加的音素距离
    ref_idx = 0
    for ref_sym, hyp_sym in pairs:
        if not ref_sym:                       # 插入：没有参考音素
            continue
        if ref_idx < n_ref:
            wi = word_of[ref_idx]
            if hyp_sym:
                heard[wi].append(hyp_sym)
                cost = _substitution_cost(ref_sym, hyp_sym)
                if cost < RECALL_MAX_DIST:
                    recalled[wi] += 1
            else:
                cost = 1.0                    # 删除：这个音素完全没发出来
            dist_sum[wi] += cost
        ref_idx += 1

    word_dist = [
        dist_sum[wi] / len(groups[wi]) if groups[wi] else 0.0
        for wi in range(len(groups))
    ]
    return recalled, heard, word_dist

def _word_level(word_avg: float, good: float, bad: float) -> str:
    """把一个词的平均音素距离分成 "good" / "ok" / "bad"。

    放在音素质量分用的同一 [good, bad] 窗口上，三色高亮才跟分桶走：
    ``frac`` 是该词从完美朗读（0）到「完全错」（1）的位置。
    分界来自模型校准（``word_good_frac`` / ``word_bad_frac``）；默认大约三等分。
    """
    span = max(bad - good, BAD_MIN_SPAN)
    frac = (word_avg - good) / span
    if frac <= WORD_GOOD_FRAC:
        return "good"
    if frac >= WORD_BAD_FRAC:
        return "bad"
    return "ok"

# IPA「[我的]」播放要求用户切片最短时长（秒）。更短的窗口当作不可信对齐，不给播。
IPA_CLIP_MIN_SEC = 0.05

def ipa_clip_trusted(start_sec: float | None, end_sec: float | None) -> bool:
    """词的带时对齐跨度够长、可以回放时为 True。

    功能：判断 start/end 是否都有值且时长 >= IPA_CLIP_MIN_SEC。
    参数：
        start_sec: 起点秒；未知则为 None。
        end_sec: 终点秒；未知则为 None。
    返回：是否可信可播。
    """
    if start_sec is None or end_sec is None:
        return False
    return (float(end_sec) - float(start_sec)) >= IPA_CLIP_MIN_SEC

def build_ipa_words(tokens: list[str],
                    groups: list[list[str]],
                    pairs: list[tuple[str, str]],
                    spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """逐词 IPA 对齐，外加可选的用户音频时间跨度，给 UI 面板。

    功能：把音素对齐拆回每个显示词，并附上可播的 start_sec/end_sec。
    参数：
        tokens: ``expected_text.split()`` 得到的显示词。
        groups: 每个词的参考音素组。
        pairs: 整句音素对齐。
        spans: 与拍平后的识别音素序列平行（顺序与非空 ``hyp`` 侧相同）。
    返回：每词一个 dict（word / expected / heard / ok / start_sec / end_sec）。

    插入留在当前词里（上一个参考音素所属的词，句首则是词 0），后面的词不会被挤错位。
    """
    n_words = len(groups)
    if n_words == 0:
        return []

    word_of: list[int] = []
    for wi, group in enumerate(groups):
        word_of.extend([wi] * len(group))

    expected: list[list[str]] = [[] for _ in range(n_words)]
    heard: list[list[str]] = [[] for _ in range(n_words)]
    ok: list[list[bool]] = [[] for _ in range(n_words)]
    times: list[list[tuple[float, float]]] = [[] for _ in range(n_words)]

    ref_idx = 0
    spoken_idx = 0
    last_word = 0

    for ref_sym, hyp_sym in pairs:
        if not ref_sym:
            wi = last_word if ref_idx > 0 else 0
            wi = min(max(wi, 0), n_words - 1)
            expected[wi].append("")
            heard[wi].append(hyp_sym)
            ok[wi].append(False)
            if hyp_sym and spoken_idx < len(spans):
                sp = spans[spoken_idx]
                times[wi].append((float(sp["t0"]), float(sp["t1"])))
                spoken_idx += 1
            continue

        wi = word_of[ref_idx] if ref_idx < len(word_of) else n_words - 1
        last_word = wi
        expected[wi].append(ref_sym)
        heard[wi].append(hyp_sym)
        if hyp_sym:
            ok[wi].append(_substitution_cost(ref_sym, hyp_sym) < RECALL_MAX_DIST)
            if spoken_idx < len(spans):
                sp = spans[spoken_idx]
                times[wi].append((float(sp["t0"]), float(sp["t1"])))
                spoken_idx += 1
        else:
            ok[wi].append(False)
        ref_idx += 1

    out: list[dict[str, Any]] = []
    for wi in range(n_words):
        start_sec: float | None = None
        end_sec: float | None = None
        if times[wi]:
            start_sec = min(t0 for t0, _t1 in times[wi])
            end_sec = max(t1 for _t0, t1 in times[wi])
            if not ipa_clip_trusted(start_sec, end_sec):
                start_sec, end_sec = None, None
        out.append({
            "word": tokens[wi] if wi < len(tokens) else "",
            "expected": expected[wi],
            "heard": heard[wi],
            "ok": ok[wi],
            "start_sec": start_sec,
            "end_sec": end_sec,
        })
    return out

def _overproduction_penalty(n_reference: int, n_spoken: int) -> float:
    """说出的音素远多于参考时的惩罚，落在 [0, 1]。

    召回和最小代价对齐都不管多出来的话，很长、完全不同的句子会给几乎每个参考音素
    挑一个接近匹配，于是全绿、分数虚高。这里量的是识别长度超出参考长度容差带多少：
    带内为 0（正常/带口音的朗读不动），过量越大越接近 1。
    OVERPRODUCTION_STRENGTH == 0 则关闭。
    """
    if OVERPRODUCTION_STRENGTH <= 0 or n_reference <= 0:
        return 0.0
    excess = (n_spoken - n_reference * (1.0 + OVERPRODUCTION_TOLERANCE)) / n_reference
    if excess <= 0.0:
        return 0.0
    return min(1.0, OVERPRODUCTION_STRENGTH * excess)

def _penalised_word_distance(word_avg: float, penalty: float, bad: float) -> float:
    """按 ``penalty`` 把词的平均距离向 ``bad`` 锚点插值。

    让三色与带惩罚的分数一致：penalty 0 时距离不变，penalty 1 时每个词都读成
    ``bad``（红），虚长/说错句的句子不再涂成绿。已经超过 ``bad`` 的词保持原样。
    """
    return word_avg + penalty * max(0.0, bad - word_avg)

def _reference_word_tags(tokens: list[str], levels: list[str]) -> list[dict[str, Any]]:
    """每个目标 token 一项 {"word", "level", "correct"}（大小写和标点保留）。

    ``level`` 驱动 GUI 三色（good/ok/bad）；``correct`` 留给仍读布尔值的调用方
    （词不是 bad 就算 "correct"）。
    """
    tags: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        level = levels[i] if i < len(levels) else "good"
        tags.append({"word": token, "level": level, "correct": level != "bad"})
    return tags

# =====================================================================
# 0–5 分桶：原始 0–100 分 -> 人校准过的档 + 百分数。
# =====================================================================
def _score_to_bucket(score: float) -> int:
    """粗 0–5 档：分数跨过了几道升序门槛（0..5）。

    没有配置 cutpoints 时返回 -1，告诉调用方继续用原始 0–100 分（及其阈值），不要用桶。
    """
    if not BUCKET_CUTPOINTS:
        return -1
    return int(sum(1 for c in BUCKET_CUTPOINTS if score >= c))

def _bucket_to_percent(bucket: int, fallback: float) -> float:
    """给用户看的百分数：该桶 [lo, hi] 区间的中点。

    取中点让同一桶里每次 take 显示同一个数（平坦），桶内引擎噪声被藏起来。
    该桶没有配置区间时退回 ``fallback``（原始分数）。
    """
    band = BUCKET_TO_PERCENT.get(str(bucket))
    if not band or len(band) != 2:
        return fallback
    lo, hi = band
    return round((float(lo) + float(hi)) / 2.0, 1)

def _grade_for_score(bucket: int, score: float) -> tuple:
    """带 +/- 深浅的 0–5 档：``score`` 落在 ``bucket`` 里时例如 ("4+", 4.33)。

    标签是桶号加上该桶原始分区间的三分位修饰（下三分之一 "-"，中间无修饰，上三分之一 "+"），
    用户能看到桶内走动，又没有百分数那种假精度。
    数值是同一标记落在连续 0–5 轴上（bucket + (third-1)/3），供会话平均和趋势比较。
    ``bucket`` 不是已配置的桶时返回 ("", -1.0)（无校准，或被 patch 过的 cutpoint 列表）。
    """
    if bucket < 0 or bucket > len(BUCKET_CUTPOINTS) or not BUCKET_CUTPOINTS:
        return "", -1.0
    edges = [0.0] + [float(c) for c in BUCKET_CUTPOINTS] + [100.0]
    lo, hi = edges[bucket], edges[bucket + 1]
    frac = (score - lo) / (hi - lo) if hi > lo else 0.5
    third = min(2, max(0, int(frac * 3)))
    label = f"{bucket}{('-', '', '+')[third]}"
    return label, round(bucket + (third - 1) / 3.0, 2)

# =====================================================================
# 入口。
# =====================================================================
def analyze(user_audio: np.ndarray,
            expected_text: str,
            reference_audio: np.ndarray | None = None,
            user_sr: int = TARGET_SAMPLE_RATE,
            reference_sr: int = KOKORO_SAMPLE_RATE,
            voice: str | None = None,
            is_reference: bool = False,
            expected_ipa: str | None = None) -> PronunciationResult:
    """在音素层面比较用户朗读与期望句子。

    功能：文本经 espeak 得参考音素，用户 wav 经 wav2vec2 得识别音素，再做特征加权
    编辑距离，返回分数、逐词/逐音素标记和转写。
    参数：
        user_audio: 用户录音波形（一维 float32）。
        expected_text: 用户被要求跟读的参考句子。
        reference_audio: Kokoro 合成的参考波形。ceiling 模式用来把该句完美朗读锚定到
            100 分；可选（缺失则退回全局 GOOD 锚点）。``np.ndarray | None`` = 数组或没有。
        user_sr: ``user_audio`` 的采样率（录音路径是 16 kHz）。
        reference_sr: ``reference_audio`` 的采样率（Kokoro 是 24 kHz）。
        voice: 合成参考时用的 Kokoro 音色（记入日志）。
        is_reference: 标记这是参考自测。现在收下是为了签名稳定；这里的诚实打分逻辑不变。
    返回：
        PronunciationResult，含分数、逐词/逐音素标记和转写。
        ``prosody`` 留空；主机从原始波形自己填。
    """
    cfg = get_config()
    _ensure_loaded()
    _ensure_calibration()

    # 从文本得到参考音素（按词分组 -> 拍平序列）。
    # Isolated IPA skips G2P: phonemizer reads "ɪ" as the letter name.
    if expected_ipa:
        phones = _normalize_phones([expected_ipa.strip()])
        if not phones:
            raise ValueError(f"empty expected ipa: {expected_ipa!r}")
        groups = [phones]
    else:
        groups = reference_word_phonemes(expected_text, cfg.espeak_language)
    reference = [p for group in groups for p in group]
    if not reference:
        raise ValueError(f"espeak produced no phonemes for: {expected_text!r}")

    spoken_spans = _spoken_timed_from_wav(_prepare_waveform(user_audio, user_sr), cfg.device)
    spoken = [span["phone"] for span in spoken_spans]

    # Ceiling 模式的 GOOD 锚点：参考 take 自己的逐音素距离，该句完美朗读映射到 100，
    # 不受「这句本身难不难」影响。
    good = _ceiling_good(reference, reference_audio, reference_sr, cfg)
    result = align_and_score(reference, spoken, good=good)

    # 把音素错误映射回整词，给 GUI 高亮。逐词平均距离用分数同一套 [good, bad] 窗口分类，
    # 三色（good/ok/bad）跟分桶走，而不是宽松的召回。
    tokens = expected_text.split()
    _recalled, heard, word_dist = _word_recall(groups, result.pairs)
    ipa_words = build_ipa_words(tokens, groups, result.pairs, spoken_spans)
    # 与分数同一套过量产出惩罚：把每个词的距离往 bad 锚点推，很长、完全不同的句子
    # 会涂成红，而不是靠碰巧匹配涂成全绿。
    overprod = _overproduction_penalty(len(reference), len(spoken))
    levels = [
        _word_level(_penalised_word_distance(word_dist[wi], overprod, result.bad_baseline),
                    result.good, result.bad_baseline)
        for wi in range(len(groups))
    ]
    # 只有 "bad" 才算错误（红，「你需要把 X 读好」）；"ok" 词可接受（浅灰），不进错误列表。
    words_with_errors = [
        tokens[wi] for wi in range(min(len(groups), len(tokens))) if levels[wi] == "bad"
    ]
    reference_words = _reference_word_tags(tokens, levels)
    word_errors = [
        {
            "word": tokens[wi] if wi < len(tokens) else "",
            "expected": groups[wi],
            "heard": heard[wi],
            "level": levels[wi],
            "distance": round(word_dist[wi], 4),
        }
        for wi in range(len(groups))
        if levels[wi] == "bad"
    ]
    word_diff = [{"expected": w} for w in words_with_errors]

    # 「听到」行：按顺序的识别音素，每个标正确/不正确。
    recognized_units = [
        {
            "unit": hyp_sym,
            "correct": bool(ref_sym) and _substitution_cost(ref_sym, hyp_sym) < RECALL_MAX_DIST,
        }
        for ref_sym, hyp_sym in result.pairs
        if hyp_sym
    ]

    transcription = " ".join(spoken)

    # 把原始 0–100 分粗化成 0–5 桶。配置了分桶时，「通过」和用户百分数来自桶；
    # 否则退回原始分数及其阈值（没有校准也不会崩）。
    bucket = _score_to_bucket(result.score)
    if bucket >= 0:
        passed = bucket >= PASS_BUCKET
        user_percent = _bucket_to_percent(bucket, result.score)
        grade, grade_value = _grade_for_score(bucket, result.score)
    else:
        passed = result.score >= cfg.score_threshold
        user_percent = result.score
        grade, grade_value = "", -1.0
    feedback = _build_feedback(result.score, passed, words_with_errors)

    logging.info(
        "[phoneme] score=%.1f -> bucket=%d grade=%s (%.0f%%) | (phoneme=%.1f recall=%.2f) | "
        "dist/phone=%.4f (good=%.3f bad=%.3f) | ref=%d spoken=%d overprod=%.2f | is_ref=%s | voice=%s | "
        "bad_words=%s | ref_ipa=%r | asr_ipa=%r",
        result.score, bucket, grade or "-", user_percent, result.phoneme_score, result.recall,
        result.per_phone_distance, result.good, result.bad_baseline,
        len(reference), len(spoken), overprod, is_reference, voice,
        words_with_errors, " ".join(reference), transcription,
    )

    # 校准 / 分析样本日志（尽力而为；对应声学引擎的 acoustic_samples.jsonl）。
    # phoneme/calibrate.py 用这里的好的*真实*尝试重新锚定 PHONEME_GOOD
    # （会丢掉 is_reference 自测，那种 ~0 距离会把锚点搞坏）；逐词块和两串音素
    # 方便事后检查低分。
    _append_sample({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "lang": _lang_key(cfg.espeak_language),
        "text": expected_text,
        "user_name": cfg.user_name,
        "voice": voice,
        "is_reference": bool(is_reference),
        "score": result.score,
        "bucket": bucket,
        "grade": grade,
        "user_percent": user_percent,
        "passed": passed,
        "phoneme_score": round(result.phoneme_score, 1),
        "recall": round(result.recall, 4),
        "per_phone_distance": round(result.per_phone_distance, 5),
        "bad_baseline": round(result.bad_baseline, 5),
        "good_anchor": round(result.good, 5),
        "n_reference": len(reference),
        "n_spoken": len(spoken),
        "reference_phonemes": reference,
        "spoken_phonemes": spoken,
        # 分数背后精确的音素对音素对齐，直接来自 align_and_score()
        # （与 _weak_phonemes 相同的 "" 空位约定）：每对是 [参考音素, 听到的音素]，
        # "" 表示删除（参考音素没发出）或插入（多出来的音素没有参考对应）。
        # 与下面逐词 expected/heard 不同，长度不一致时位置仍然精确，
        # 可直接做替换/删除/插入混淆矩阵，不必从日志再推对齐。
        "alignment": [[ref_sym, hyp_sym] for ref_sym, hyp_sym in result.pairs],
        "words": [
            {"word": tokens[wi] if wi < len(tokens) else "",
             "level": levels[wi],
             "distance": round(word_dist[wi], 4),
             "expected": groups[wi],
             "heard": heard[wi]}
            for wi in range(len(groups))
        ],
    })

    return PronunciationResult(
        score=result.score,
        word_errors=word_errors,
        prosody={},
        transcription=transcription,
        passed=passed,
        feedback=feedback,
        words_with_errors=words_with_errors,
        expected_phonemes=reference,
        transcribed_phonemes=spoken,
        word_diff=word_diff,
        reference_words=reference_words,
        recognized_units=recognized_units,
        weak_phonemes=_weak_phonemes(result.pairs),
        ipa_words=ipa_words,
        bucket=bucket,
        user_percent=user_percent,
        grade=grade,
        grade_value=grade_value,
        per_phone_distance=result.per_phone_distance,
        bad_baseline=result.bad_baseline,
        phoneme_score=result.phoneme_score,
        recall=result.recall,
        good_anchor=result.good,
    )

def _ceiling_good(reference: list[str], reference_audio: np.ndarray | None,
                  reference_sr: int, cfg) -> float | None:
    """来自参考 take 的逐句 GOOD 锚点；global 模式则为 None。

    有参考 take 时返回它的逐音素距离（ceiling 模式）；否则 None，打分器继续用全局
    PHONEME_GOOD。缺少参考绝不会让分析失败。
    """
    if cfg.good_mode != "ceiling" or reference_audio is None:
        return None
    ref_array = np.asarray(reference_audio, dtype=np.float32)
    if ref_array.size == 0:
        return None
    ceiling_spoken = _recognize_reference(ref_array, reference_sr, cfg.device)
    if not ceiling_spoken:
        return None
    return align_and_score(reference, ceiling_spoken).per_phone_distance

def _build_feedback(score: float, passed: bool, words_with_errors: list[str]) -> str:
    """给人看的短摘要，风格与声学引擎一致。"""
    lines = [f"Score: {score:.0f}/100 " + ("(passed)" if passed else "(try again)")]
    if words_with_errors:
        lines.append("❌ You need to better pronounce these words: "
                     + ", ".join(words_with_errors))
    elif passed:
        lines.append("✅ Great pronunciation!")
    return "\n".join(lines)
