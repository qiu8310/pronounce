# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""发音分析核心（声学引擎）。

从 OpenPronounce（https://github.com/Halleck45/OpenPronounce，MIT 许可）改编而来。
声学 / 音素对比逻辑作为库复用；原项目的 Web 前端和内置 TTS 已去掉。
在 Mimora 里，参考音频由现有的 Kokoro TTS 生成，再以 NumPy 数组传入
``analyze``，因此本模块自己不合成语音，也不接触 GUI。

本引擎在做什么：用 Wav2Vec2 从用户录音和参考音频各自抽出一段「嵌入」
（embedding，每一步语音的特征向量），再用 DTW（动态时间规整）对齐两条
序列，得到声学分数；同时用 CTC 头把用户音频转成文字，再和目标句子做
音素 / 字符对比。最终把声学、音素、词级三路合成 0–100 分。

对外接口：
    load_models()  -- 只加载一次 Wav2Vec2 权重（模式启动时、在线程里调用）。
    warm_up()      -- 跑一遍空数据，去掉首次推理的额外延迟（和 tts.py 同一套路）。
    analyze(...)   -- 唯一入口，返回 PronunciationResult。

设计说明：
    * 模型在第一次真正用到时才懒加载；``load_models`` 只是把这件事变得显式，
      好让沉重的下载 / 初始化可以放到后台守护线程里，和 ``tts.py`` 的预热方式一致。
    * 设置（模型、设备、口音、阈值、日志目录、用户名）来自本库自己的
      ``AnalyzerConfig``（见 acoustic/config.py），在「用到时」再读。
      宿主通过 ``acoustic.configure(...)`` 注入一次即可；默认值让本包能独立运行，
      因此永远不会 import 宿主应用。
    * Wav2Vec2 需要 16 kHz 单声道。用户录音在采集路径上已经是 16 kHz；
      Kokoro 参考音是 24 kHz，所以在这里重采样。
"""

import json
import logging
import re
import threading
from datetime import datetime
# lru_cache：给函数加「最近最少使用」结果缓存的装饰器，下面 _phonemize_word 会用到。
from functools import lru_cache
from pathlib import Path
# Any：类型标注里的「任意类型」，表示这个位置不限制具体类型。
from typing import Any

import Levenshtein
import numpy as np
import torch
from fastdtw import fastdtw  # pyright: ignore[reportAttributeAccessIssue]
from phonemizer import phonemize
from scipy.spatial.distance import cosine
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor  # pyright: ignore[reportAttributeAccessIssue]

# 波形预处理集中在 pronounce.common（一份不依赖 torch 的实现）。
# 下面的 as 别名保留本模块惯用的本地名（以及测这些名字的单元测试）。
# 「as _xxx」：导入时换成带下划线的内部名，表示只给本文件用。
from pronounce.common.audio import (
    TARGET_SAMPLE_RATE,
    prepare_waveform as _prepare_waveform,
    trim_silence as _trim_silence,
    waveform_digest,
)

# 打包的 espeak-ng 注册，和音素引擎共用。本引擎以前自己没有注册，
# 只能碰到进程里碰巧已有的 espeak——系统安装，或 Kokoro/misaki 导入时的副作用，
# 两者都不保证存在。详见该模块。
from pronounce.common.espeak import ensure_espeak

# 设置来自本库自己的 AnalyzerConfig（见 acoustic/config.py），从不读宿主应用：
# 宿主在启动时通过 acoustic.configure() 注入一次。get_config() 返回当前生效的配置。
from .config import get_config

# =====================================================================
# 配置。
#
# 所有宿主可调的设置（模型、设备、espeak 口音、分数阈值、声学下限默认值、
# 日志目录、用户名）都来自 get_config() 拿到的 AnalyzerConfig；在用到时才读，
# 这样宿主可以在 import 之后、分析之前用 acoustic.configure() 注入。
# 下面这些常量是分析器固有的，宿主调不了。
# =====================================================================
# Wav2Vec2 严格要求 16 kHz 单声道输入：TARGET_SAMPLE_RATE
# （上面从 pronounce.common.audio 导入）。
# Kokoro 以 24 kHz 合成；仅作为「单独用本引擎」时参考音的默认采样率。
# Mimora 的 main.py 总会显式传入 reference_sr（当前 TTS 后端的原生速率），
# 所以那里从不会用到这个默认值。（本引擎只做英语，英语走 Kokoro，
# 因此这个数字也和 Mimora 实际的英语参考音采样率一致。）
# 24_000 里的下划线只是方便读数，值等于 24000。
KOKORO_SAMPLE_RATE = 24_000

# ---------------------------------------------------------------------
# 声学分数校准。
#
# 声学分量比较的是两条 Wav2Vec2 嵌入序列之间、逐步的余弦 DTW 距离。
# 两个锚点把它映射到 0–100：
#   * floor（声学「好」距离）——另一说话人一次「好」尝试的典型逐步距离
#     （用户 vs TTS 音色永远到不了 0）。默认值在 AnalyzerConfig.acoustic_good，
#     可按音色 / 麦克风用 ``python acoustic/calibrate.py`` 校准，写入
#     calibration.json。current_acoustic_floor() 返回当前生效的值。
#   * ceiling——内容对不上时的逐步距离。按每句自动从「随机配对基线」
#     （两段录音未对齐帧之间的平均距离）推出来，所以能适应每句短语，
#     不用手调。
# ---------------------------------------------------------------------
ACOUSTIC_BAD_DEFAULT = 0.60      # 没有按句基线时用的固定上限
ACOUSTIC_BAD_FRACTION = 0.9      # 上限 = 随机配对基线的这个比例
ACOUSTIC_MIN_SPAN = 0.05         # 下限到上限的最小间距（避免刻度退化成一条线）

# 持久化校准（覆盖下限）的独立运行默认路径：就在本模块旁边，
# 无宿主的调用方和评测工具都按这个位置找。
# 宿主可通过 AnalyzerConfig.calibration_file 覆盖——见
# current_calibration_file()，下面所有读写都走它。
# Path(__file__).resolve().parent：
#   __file__ 是本文件路径；resolve() 变成绝对路径并解开符号链接；
#   .parent 是本文件所在目录。
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"

def current_calibration_file() -> Path:
    """当前生效的校准文件路径：宿主指定的，否则用本包默认路径。

    每次读写都重新问，而不是 import 时算死一次。因为本模块被导入时
    configure() 可能还没跑，答案必须反映「此刻」的配置。

    功能：给出本次该用的 calibration.json 路径。
    返回：pathlib.Path。``-> Path`` 是返回值的类型标注。
    """
    # 「x or y」：左边为假（None / 空）时用右边。这里表示「有宿主路径就用，否则默认」。
    return get_config().calibration_file or CALIBRATION_FILE

def samples_file() -> Path:
    """每次尝试写入的校准样本日志路径。

    放在配置的日志目录下；``acoustic/calibrate.py`` 会读它，
    以便按需重新计算声学下限。

    功能：拼出 acoustic_samples.jsonl 的完整路径。
    返回：Path。
    """
    return Path(get_config().log_dir) / "acoustic_samples.jsonl"

# 声学下限是按练习用户分的：calibration.json 把用户名
# （AnalyzerConfig.user_name，未设置时为 ""）映射到该用户的下限，
# 放在 "users" 对象下：
#   {"users": {"": {"acoustic_good": 0.18, "created": ...}, "valery": {...}}}
# 更早的、尚未按用户拆开的文件把下限放在顶层（{"acoustic_good": ...}）；
# 仍当作回退读取，下次保存时迁进 "" 这个默认档案。
def _load_calibration() -> float:
    """返回当前用户已校准的声学下限；没有则用配置默认值。"""
    user_name = get_config().user_name
    default = get_config().acoustic_good
    path = current_calibration_file()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            users = data.get("users") if isinstance(data, dict) else None
            entry = users.get(user_name) if isinstance(users, dict) else None
            if isinstance(entry, dict) and "acoustic_good" in entry:
                value = float(entry["acoustic_good"])
                source = f"user={user_name!r}"  # !r 表示用 repr()，字符串会带引号，方便日志里辨认空用户名
            elif isinstance(data, dict) and "acoustic_good" in data:
                value = float(data["acoustic_good"])  # 旧版扁平文件（下限在顶层）
                source = "legacy"
            else:
                return default
            logging.info(f"[acoustic] Loaded calibration ({source}): "
                         f"acoustic_good={value:.4f} ({path})")
            return value
    except Exception:
        logging.exception("Failed to read calibration file; using defaults:")
    return default

# 当前生效的已校准声学下限，首次加载后缓存，calibration.json 只读一次。
# 懒加载（好让 configure() 先装好用户名），活跃用户一变会自动刷新；
# None 表示「还没加载」。
# float | None：类型可以是 float，也可以是 None（联合类型，读作「或」）。
_acoustic_good: float | None = None
_acoustic_good_user: str | None = None

def current_acoustic_floor() -> float:
    """返回当前用户已校准的声学下限（带缓存）。

    配置里的用户一变就从 calibration.json 重新加载，所以用 configure()
    切换用户时能拿到对的下限，不必显式复位。

    功能：给出打分公式里用的声学 floor。
    返回：float，越小表示「好的尝试」离参考越近。
    """
    # global：下面要给模块级变量赋值。不写 global 的话，赋值会变成函数里的局部变量。
    global _acoustic_good, _acoustic_good_user
    user_name = get_config().user_name
    if _acoustic_good is None or _acoustic_good_user != user_name:
        _acoustic_good = _load_calibration()
        _acoustic_good_user = user_name
    return _acoustic_good

# 把旧版扁平校准迁进某个用户档案时，要一并带走的键。
_LEGACY_CALIBRATION_KEYS = ("acoustic_good", "created", "samples_used", "voice")

def save_calibration(acoustic_good: float, extra: dict[str, Any] | None = None) -> None:
    """把当前用户的声学下限写入文件，并立刻用到本进程。

    下限存在 ``users`` 映射里、当前用户名下面，其他用户的校准不动。

    功能：持久化并激活一条用户校准。
    参数：
        acoustic_good: 该用户的声学下限。
        extra: 可选的额外字段字典。``dict[str, Any] | None`` 表示
            「键为 str、值为任意类型的字典，或 None」；默认 None 表示没有额外字段。
    返回：无（``-> None``）。
    """
    global _acoustic_good, _acoustic_good_user
    user_name = get_config().user_name
    # 只解析一次：本函数又读又写，即使中途配置被换掉，也必须是同一个文件。
    path = current_calibration_file()
    entry: dict[str, Any] = {
        "acoustic_good": round(float(acoustic_good), 5),
        "created": datetime.now().isoformat(timespec="seconds"),
        "user_name": user_name,
    }
    if extra:
        entry.update(extra)

    # 合并进已有仓库，好让其他用户的下限还在。
    data: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                data = existing
        except Exception:
            logging.exception("Calibration file unreadable; rewriting it:")
    users = data.get("users")
    if not isinstance(users, dict):
        users = {}
    # 把「按用户拆分之前」的下限停在默认（""）档案里，避免弄丢。
    # 下面是字典推导：只复制旧文件里仍存在的那些遗留键。
    if "acoustic_good" in data and "" not in users:
        users[""] = {k: data[k] for k in _LEGACY_CALIBRATION_KEYS if k in data}
    users[user_name] = entry

    # 宿主的目录在 import 时由 config.py 建好；独立调用方可能把路径指到新地方。
    # parents=True：缺哪一层目录都建；exist_ok=True：目录已存在也不报错。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"users": users}, indent=2) + "\n", encoding="utf-8")
    _acoustic_good = float(acoustic_good)
    _acoustic_good_user = user_name
    logging.info(f"[acoustic] Saved calibration user={user_name!r} "
                 f"acoustic_good={acoustic_good:.4f} -> {path}")

# 懒初始化的模型单例（加载一次，之后每次分析都复用）。
# 三个变量同样是「有对象 | 还没加载」的联合类型。
_processor: Wav2Vec2Processor | None = None
_model: Wav2Vec2Model | None = None          # 嵌入；其实是 _model_ctc.wav2vec2 的别名
_model_ctc: Wav2Vec2ForCTC | None = None     # 转写（识别出了什么字）
# 锁住 load_models()，避免并发调用把权重加载两遍
# （对外 API 并不要求调用方自己串行化）。
# threading.Lock()：互斥锁，同一时刻只让一个线程进入 with 块。
_load_lock = threading.Lock()

# =====================================================================
# 结果类型（本模块对 GUI 层的契约）
# =====================================================================
# PronunciationResult 是引擎无关的共享类型（pronounce.common）：
# 本声学引擎和 pronounce.score.phoneme 返回同一形状，GUI 才不用管是哪个引擎。
# 经本包 __init__ 再导出，所以 ``pronounce.score.acoustic.PronunciationResult``
# 仍然能用。本引擎填充 ``acoustic_*`` 字段；音素专用字段保持默认值。
from pronounce.common import PronunciationResult


# =====================================================================
# 模型生命周期
# =====================================================================
def load_models() -> None:
    """把 Wav2Vec2 权重复入内存一次。重复调用是安全的。

    很重（首次约 1.2 GB 下载）；模式启动时从后台守护线程调用，GUI 才不会卡住。

    功能：确保处理器、CTC 模型和嵌入编码器都已加载。
    返回：无。
    """
    global _processor, _model, _model_ctc
    # with 锁：进入块时加锁，离开时（包括中途出错）自动解锁。
    with _load_lock:
        if _model is not None and _model_ctc is not None and _processor is not None:
            return

        # 函数内部才 import：懒导入。没真正加载模型时，不会去碰兼容补丁模块。
        from pronounce.common.compat import allow_torch_load_for_trusted_models

        # 这里也注册一遍（_phonemize_word 里还会再调）：好让日志出现在启动阶段、
        # 和其他加载消息挨在一起，而不是夹在第一次分析中间。调用是幂等的，第二次几乎免费。
        ensure_espeak()
        allow_torch_load_for_trusted_models()
        cfg = get_config()
        _processor = Wav2Vec2Processor.from_pretrained(cfg.model_name)

        _model_ctc = Wav2Vec2ForCTC.from_pretrained(cfg.model_name).to(cfg.device)
        _model_ctc.eval()

        # CTC 检查点里已经带完整的基座编码器——复用它来抽嵌入，
        # 就不必再从磁盘加载一份完全相同的权重（约 1.2 GB 内存 / 显存）。
        _model = _model_ctc.wav2vec2

def warm_up() -> None:
    """对两个模型跑一段很短的空数据，去掉首次调用的额外延迟。

    功能：预热推理路径，让用户第一次真实打分不再多等一次编译 / 分配。
    返回：无。
    """
    load_models()
    # // 是整除（向下取整）。采样率的一半个样本 ≈ 0.5 秒静音。
    dummy = np.zeros(TARGET_SAMPLE_RATE // 2, dtype=np.float32)  # 0.5 秒静音
    extract_embeddings(dummy)
    transcribe(dummy)

def _ensure_loaded() -> None:
    """推理前保证模型已在内存里。"""
    if _model is None or _model_ctc is None or _processor is None:
        load_models()

# 音频预处理在 pronounce.common.audio（上面导入了 _prepare_waveform /
# _trim_silence），和音素引擎、宿主的韵律层共用，大家量的是同一份整理后的信号。

# =====================================================================
# Wav2Vec2 推理
# =====================================================================
def extract_embeddings(audio_waveform: np.ndarray,
                       sampling_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """抽出原始 Wav2Vec2 嵌入，形状为 (时间步, 特征维)。

    功能：把一段波形变成按时间排列的特征向量，供后面 DTW 比对。
    参数：
        audio_waveform: 一维波形。
        sampling_rate: 采样率；默认 16 kHz（Wav2Vec2 的要求）。
    返回：二维 numpy 数组，行是时间步。
    """
    _ensure_loaded()

    inputs = _processor(audio_waveform, sampling_rate=sampling_rate,
                        return_tensors="pt", padding=True)

    input_values = inputs.input_values
    if input_values.dim() > 2:  # 去掉多出来的无用前导维度
        input_values = input_values.squeeze(0)
    input_values = input_values.to(get_config().device)

    # torch.no_grad()：上下文管理器，关掉梯度。推理不需要反传，省内存、也更快。
    with torch.no_grad():
        features = _model(input_values).last_hidden_state  # (batch, 时间, 特征)

    return features.squeeze(0).cpu().numpy()

def transcribe(audio_waveform: np.ndarray) -> str:
    """用 Wav2Vec2 的 CTC 头把音频转成文字。

    功能：得到 ASR 识别结果，供后面和目标句子做音素 / 词对比。
    参数：
        audio_waveform: 已整理到 16 kHz 的波形。
    返回：识别出的字符串（尚未做 clean_transcription）。
    """
    _ensure_loaded()

    inputs = _processor(audio_waveform, sampling_rate=TARGET_SAMPLE_RATE,
                        return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(get_config().device)

    with torch.no_grad():
        logits = _model_ctc(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1).cpu()  # 在 CPU 上解码
    return _processor.batch_decode(predicted_ids)[0]

# =====================================================================
# 音素 / 文本对比（复用的 OpenPronounce 核心）
# =====================================================================
# @lru_cache：把函数结果按参数缓存起来。maxsize=4096 表示最多记 4096 条；
# 满了就丢掉最久没用的（Least Recently Used）。phonemize 每次会拉起 espeak，很贵。
@lru_cache(maxsize=4096)
def _phonemize_word(word: str) -> tuple:
    """把一个词转成音素（每次 phonemize 都会拉起 espeak，所以缓存结果；
    同一词既会在同一句的多次尝试里重复，也会跨句子重复）。

    缓存键只有这个词，尽管结果还依赖 espeak 口音：configure() 在语言
    改变时会清这个缓存，所以过期口音的条目永远不会被拿出来用。
    """
    # 真正扛事的那次调用，不是 load_models() 里那次：get_word_phonemes()
    # 可以在从不加载识别器的情况下被走到；本引擎直到现在才自己注册——
    # 以前只有进程里别人（系统安装，或导入 Kokoro）已经提供过 espeak 才管用。
    ensure_espeak()
    try:
        return tuple(phonemize(word, language=get_config().espeak_language,
                               backend="espeak",
                               strip=True, preserve_punctuation=False).split())
    except Exception:
        # 损坏 / 缺失的 espeak-ng 不能让整次分析失败，但也不能悄悄吞掉：
        # 音素为空时 compare_transcriptions 会跳过每个词，分数会在无声中被扭曲。
        logging.exception(f"[acoustic] espeak phonemization failed for "
                          f"{word!r}; trying festival")
        try:
            return tuple(phonemize(word, language="en-us", backend="festival",
                                   strip=True, preserve_punctuation=False).split())
        except Exception:
            logging.exception(f"[acoustic] festival fallback failed for "
                              f"{word!r}; returning no phonemes - the word "
                              f"will be excluded from scoring")
            return ()  # 所有后端都失败时的回退

def get_word_phonemes(text: str) -> list[tuple]:
    """按出现顺序返回每个词的 (词, 音素) 对。

    功能：把一句文本拆成词，并给每个词标音素。
    参数：
        text: 原始或已清洗的文本。
    返回：``list[tuple]``，即元组的列表；每个元组是 (单词, 音素序列)。
    """
    # 按词切，忽略标点，避免 "times," 这种带逗号的词把音素搞乱。
    words = re.findall(r"\b[\w']+\b", text)
    # 列表推导：对每个 word 调一次（带缓存的）_phonemize_word。
    return [(word, _phonemize_word(word)) for word in words]

def get_phonemes_with_word_mapping(text: str):
    """返回音素列表，以及 {音素下标: 来源词} 的映射。

    功能：把整句摊成一条音素链，同时记住每个音素来自哪个词。
    参数：
        text: 文本。
    返回：二元组 (phonemes, phoneme_to_word)。
    """
    phonemes: list[str] = []
    phoneme_to_word: dict[int, str] = {}

    for word, word_phonemes in get_word_phonemes(text):
        for phoneme in word_phonemes:
            phoneme_to_word[len(phonemes)] = word
            phonemes.append(phoneme)

    return phonemes, phoneme_to_word

def compare_transcriptions(transcription: str, text_reference: str) -> dict[str, Any]:
    """把 ASR 转写和期望文本做对比。

    通过音素对齐找出每个词的发音错误，并返回打分公式要用的那些距离。

    功能：算出字符距离、音素距离、以及按词列出的错误。
    参数：
        transcription: 识别出的句子。
        text_reference: 用户本该读的句子。
    返回：含 char_distance、phoneme_distance、errors、feedback 等键的字典。
    """
    # 两边用同一套规则规范化（小写、去掉标点）。
    # 转写通常已经比较干净，但参考句还带着标点——如果不也清掉，
    # 每个逗号 / 句号都会算一次必错的编辑，把 word_error_rate 抬高
    # （短句上特别明显）。
    transcription_clean = clean_transcription(transcription)
    reference_clean = clean_transcription(text_reference)

    # 转写和参考文本之间的编辑距离。故意用「字符」级（名字也这么叫）：
    # 接近正确的词能拿部分分；若按「整词」算距离，差一点就整词全错。
    char_distance = Levenshtein.distance(transcription_clean, reference_clean)

    # 两边都抽音素。参考侧保留按词分组，后面按词边界走的时候直接复用，
    # 不必对每个词再跑一遍 espeak。
    expected_pairs = get_word_phonemes(text_reference)
    # 双重列表推导：先拆每个 (词, 该词音素)，再摊平所有音素。_word 表示这个词名用不到。
    expected_phonemes = [p for _word, word_phonemes in expected_pairs for p in word_phonemes]
    transcribed_phonemes, transcribed_map = get_phonemes_with_word_mapping(transcription_clean)

    # 全局音素距离：把音素串起来之后做「字符」级编辑距离。
    # espeak 每个词返回一段不分隔的音素串（例如 "ðə"），若对「词 token 列表」
    # 做编辑距离，整个词会变成非对即错；按字符近似单个音素，接近的能拿部分分。
    expected_join = " ".join(expected_phonemes)
    transcribed_join = " ".join(transcribed_phonemes)
    phoneme_distance = Levenshtein.distance(expected_join, transcribed_join)

    errors: list[dict[str, Any]] = []
    words_with_errors = set()

    # 每个「期望音素」下标对齐到哪些「转写音素」下标。
    # 这样才能处理 1 对 N、N 对 1（例如 "I'm" -> "I M"）。
    alignment_map = [set() for _ in range(len(expected_phonemes))]

    opcodes = Levenshtein.opcodes(expected_phonemes, transcribed_phonemes)

    # opcodes 每项是 (操作名, 期望起, 期望止, 转写起, 转写止)，一次解包成 5 个变量。
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # zip：把两个 range 一对一对地绑在一起。
            for k, l in zip(range(i1, i2), range(j1, j2)):
                alignment_map[k].add(l)
        elif tag == 'replace':
            # 按比例映射被替换的区间，而不是全对全，
            # 避免 "Hello how are" 变成「谁都对上谁」。
            len_i = i2 - i1
            len_j = j2 - j1
            for k in range(i1, i2):
                start_j = j1 + int((k - i1) * len_j / len_i)
                end_j = j1 + int((k - i1 + 1) * len_j / len_i)
                if start_j == end_j and len_j > 0:
                    idx = min(start_j, j2 - 1)
                    alignment_map[k].add(idx)
                else:
                    for l in range(start_j, end_j):
                        alignment_map[k].add(l)
        # 'delete'（缺了期望音素）和 'insert'（多了转写音素）
        # 这里不需要写进对齐表。

    # 按参考词的顺序走，才能可靠找回音素边界
    # （扁平的「音素→词」映射分不清相邻的重复词）。
    current_phoneme_idx = 0

    for word, p_list in expected_pairs:
        if not p_list:
            continue  # 这个词没产出音素（例如数字 / 符号）

        word_indices = range(current_phoneme_idx, current_phoneme_idx + len(p_list))
        current_phoneme_idx += len(p_list)

        # 收集这个词对齐到的那些转写音素下标。
        matched_trans_indices = set()
        for idx in word_indices:
            if idx < len(alignment_map):
                matched_trans_indices.update(alignment_map[idx])

        if not matched_trans_indices:
            # 转写里这个词完全失踪。
            errors.append({"position": word_indices.start, "expected": word,
                           "actual": "", "word": word})
            words_with_errors.add(word)
            continue

        sorted_trans_indices = sorted(matched_trans_indices)

        # 还原实际听到的词，去重但保持出现顺序。
        actual_words: list[str] = []
        seen_words = set()
        for tidx in sorted_trans_indices:
            if tidx in transcribed_map:
                w = transcribed_map[tidx]
                if w not in seen_words:
                    actual_words.append(w)
                    seen_words.add(w)
        actual_text = " ".join(actual_words)

        # 期望 vs 实际音素，按字符比。
        # 先拼成字符串很重要：列表里每个元素是「整词的音素串」，
        # 若对列表做编辑距离，任何一个差异都会整词判错，下面 40% 容差就没了。
        # 括号外的 `for ... in ...` 是生成器表达式：按需产出每个音素，再 join 成一条字符串。
        expected_str = "".join(expected_phonemes[i] for i in word_indices)
        actual_str = "".join(transcribed_phonemes[i] for i in sorted_trans_indices)

        p_dist = Levenshtein.distance(expected_str, actual_str)

        # 音素编辑距离超过长度的 40%，就标成发音错误。
        if p_dist > len(expected_str) * 0.4:
            errors.append({
                "position": word_indices.start,
                "expected": expected_str,           # 期望音素
                "actual": actual_str,               # 实际音素
                "word": word,                       # 期望的词面
                "actual_word": actual_text,         # 实际词面（例如 "I M"）
            })
            words_with_errors.add(word)

    # 给人看的反馈摘要。
    feedback = "🔊 Feedback on your pronunciation:\n"
    if words_with_errors:
        feedback += "❌ You need to better pronounce these words: " + ", ".join(words_with_errors) + "\n"
    else:
        feedback += "✅ Your pronunciation is excellent! 🎉\n"

    return {
        "char_distance": char_distance,
        "reference_length": len(reference_clean),
        "phoneme_distance": phoneme_distance,
        "phoneme_length": len(expected_join),
        "errors": errors,
        "feedback": feedback,
        "transcribe": transcription,
        "expected_phonemes": expected_phonemes,
        "transcribed_phonemes": transcribed_phonemes,
        "words_with_errors": list(words_with_errors),
    }

def word_level_diff(transcription: str, text_reference: str) -> list[dict[str, str]]:
    """把识别出的词和目标短语对齐，只返回对不上的那些。

    让 GUI 能展示具体的「期望 → 听到」对，而不是整段 ASR 原文。
    两边用同一套清洗（小写、去标点），再按空白切词，然后用
    ``Levenshtein.opcodes`` 做词 token 级编辑距离（音素对比用的是同一个原语，
    它接受 token 列表）：

        * 替换 -> {"expected": "time", "heard": "times"}
        * 删除 -> {"expected": "the",  "heard": ""}      （词被丢掉）
        * 插入 -> {"expected": "",      "heard": "uh"}    （多出来的词）

    'equal' 段会跳过，所以空列表表示识别和目标逐词一致。

    功能：给出词级差异列表，供界面展示。
    参数：
        transcription: ASR 文本。
        text_reference: 目标短语。
    返回：每项含 expected / heard 两个字符串的字典列表。
    """
    expected_words = clean_transcription(text_reference).split()
    heard_words = clean_transcription(transcription).split()

    diffs: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(expected_words, heard_words):
        if tag == "equal":
            continue
        diffs.append({
            "expected": " ".join(expected_words[i1:i2]),
            "heard": " ".join(heard_words[j1:j2]),
        })
    return diffs

def heard_word_tags(transcription: str, text_reference: str) -> list[dict[str, Any]]:
    """给每个识别出的词打标：是否匹配目标短语。

    按开口顺序，听到的每个词一条::

        [{"word": "hullo", "correct": False}, {"word": "i", "correct": True}, ...]

    词落在与 :func:`word_level_diff` 同一套词 token 对齐的 'equal' 段里，
    才算 ``correct``。GUI 好在原始 ASR 行上给认对的词上色。

    功能：按「听到的词」顺序标记对错。
    参数：
        transcription: ASR 文本。
        text_reference: 目标短语。
    返回：每项含 word / correct 的字典列表。
    """
    expected_words = clean_transcription(text_reference).split()
    heard_words = clean_transcription(transcription).split()

    # 列表推导：先默认每个听到的词都是错的，下面再把 equal 段改成 True。
    tags = [{"word": w, "correct": False} for w in heard_words]
    # _i1、_i2：下划线前缀表示「解包时占位，这个值用不到」。
    for tag, _i1, _i2, j1, j2 in Levenshtein.opcodes(expected_words, heard_words):
        if tag == "equal":
            for j in range(j1, j2):
                tags[j]["correct"] = True
    return tags

def reference_word_tags(text_reference: str,
                        words_with_errors: list[str]) -> list[dict[str, Any]]:
    """给目标短语的每个词打标：发得对还是不对。

    按顺序每个短语词一条，保留原始 token（大小写和标点）方便展示::

        [{"word": "Hello,", "correct": True}, {"word": "world", "correct": False}]

    规范化后的形式出现在 ``words_with_errors`` 里，这个词就不正确。
    规范化方式和 GUI 以前的行内检查一致（小写、去掉边上的标点），
    高亮才不会变。引擎无关：音素引擎也会用「音素错误映回词」做出同一形状。

    功能：按「目标句的词」顺序标记对错，供界面高亮。
    参数：
        text_reference: 目标短语（可带原始标点）。
        words_with_errors: 被判错的词（通常已是小写）。
    返回：每项含 word / correct 的字典列表。
    """
    # 集合推导：把所有错词收成小写集合，后面用「in」判断会很快。
    error_words = {w.lower() for w in words_with_errors}
    tags: list[dict[str, Any]] = []
    for token in text_reference.split():
        clean = token.lower().strip(".,!?;:\"")
        tags.append({"word": token, "correct": clean not in error_words})
    return tags

def _random_pair_baseline(emb_a: np.ndarray, emb_b: np.ndarray,
                          n_pairs: int = 2000, seed: int = 0) -> float:
    """两条嵌入里随机配对帧的平均余弦距离。

    近似「内容完全无关」时的逐步 DTW 距离，给每句自动一个「完全念错」的声学上限。
    """
    if len(emb_a) == 0 or len(emb_b) == 0:
        # 退化输入（几乎空的音频）：没有帧可抽，退回固定上限，别把分析弄崩。
        return ACOUSTIC_BAD_DEFAULT
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(emb_a), n_pairs)
    j = rng.integers(0, len(emb_b), n_pairs)
    a, b = emb_a[i], emb_b[j]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return float(np.mean(1.0 - num / den))

def acoustic_bad_for(baseline: float, acoustic_good: float | None = None) -> float:
    """由随机配对基线推出的、这一句自己的声学上限。

    功能：把「完全念错」锚在这句话的基线上，同时保证比下限至少高 ACOUSTIC_MIN_SPAN。
    参数：
        baseline: _random_pair_baseline 的结果。
        acoustic_good: 可选的下限覆盖；None 表示用当前校准下限。
    返回：这一句打分用的 ceiling。
    """
    # 三元表达式：有覆盖就用覆盖，否则读缓存里的校准下限。
    good = current_acoustic_floor() if acoustic_good is None else acoustic_good
    return max(ACOUSTIC_BAD_FRACTION * baseline, good + ACOUSTIC_MIN_SPAN)

def compute_pronunciation_score(acoustic_per_step: float,
                                phoneme_error_rate: float,
                                word_error_rate: float,
                                acoustic_bad: float | None = None,
                                acoustic_good: float | None = None) -> float:
    """把三个已归一化的分量合成 0–100 分。

    功能：声学 DTW、音素错误率、字符错误率加权求和。
    参数：
        acoustic_per_step: 每个对齐步的余弦 DTW 距离（与句子长短无关）。
        phoneme_error_rate: 音素编辑距离 / 期望音素个数。
        word_error_rate: 字符编辑距离 / 参考文本长度（历史名字仍叫 word）。
        acoustic_bad: 本句上限（见 ``acoustic_bad_for``）；没有基线时退回固定默认。
        acoustic_good: 下限覆盖；默认用当前生效的校准下限（见 ``current_acoustic_floor``）。
    返回：保留两位小数的 0–100 分数。
    """
    good = current_acoustic_floor() if acoustic_good is None else acoustic_good
    bad = ACOUSTIC_BAD_DEFAULT if acoustic_bad is None else acoustic_bad
    bad = max(bad, good + ACOUSTIC_MIN_SPAN)

    dtw_score = 100.0 * min(1.0, max(0.0, 1.0 - (acoustic_per_step - good) / (bad - good)))
    phoneme_score = 100.0 * min(1.0, max(0.0, 1.0 - phoneme_error_rate))
    word_score = 100.0 * min(1.0, max(0.0, 1.0 - word_error_rate))

    # 权重：声学 DTW 40%，音素 30%，词 / 字符 30%。
    final_score = 0.4 * dtw_score + 0.3 * phoneme_score + 0.3 * word_score

    final_score = min(100.0, max(0.0, final_score))
    return round(final_score, 2)

# 韵律（音高和能量曲线）不是本引擎的事：它属于引擎无关的
# ``mimora/prosody.py``，由宿主从原始用户 / 参考波形算出来，
# 这样每套引擎都能用同一套图。
# ``analyze`` 返回空的 ``prosody`` 字典；宿主再填进去。

def clean_transcription(text: str) -> str:
    """转写小写、去掉标点、合并空白。

    只留下 ``[a-z' ]``：数字和非 ASCII 字母会和标点一起被丢掉
    （"2" vs ASR 的 "two"，"mañana" -> "maana"），期望文本里有这些时
    会把词错误率抬高。英语 ASR 检查点本来也吐不出这些字符，所以只记日志
    不修——这样一句奇怪的分数可以在日志里追到原因。

    功能：把 ASR / 参考文本收成可比较的字母串。
    参数：
        text: 原始字符串。
    返回：只含小写字母、撇号和空格的字符串。
    """
    text = text.lower().strip()
    lost = sorted(set(re.findall(r"[0-9]|[^\x00-\x7f]", text)))
    if lost:
        logging.info("[acoustic] clean_transcription dropped %r from %r - "
                     "digits/non-ASCII are outside the scoring alphabet",
                     "".join(lost), text)
    text = re.sub(r"[^a-zA-Z' ]+", "", text)
    return " ".join(text.split()).strip()

# =====================================================================
# 参考音特征缓存
# =====================================================================
# 同一句会对着同一段 Kokoro 参考音练很多次，参考侧波形和嵌入在两次尝试之间
# 不会变。缓存最近一次参考（同一时间只练一句），重复尝试就可以跳过参考音上的
# Wav2Vec2。参考韵律另外缓存在 mimora/prosody.py（F0 / 能量曲线归它管）。
# 锁让并发的 analyze() 安全；应用里 GUI 的 is_processing_audio 已经串行化了，
# 所以这把锁通常没人抢。
_reference_cache: dict[str, Any] = {}
_reference_cache_lock = threading.Lock()

def _reference_features(reference_audio: np.ndarray, reference_sr: int) -> dict[str, Any]:
    """返回参考音整理后的波形和嵌入。"""
    global _reference_cache
    arr = np.asarray(reference_audio)
    key = (reference_sr, arr.shape, waveform_digest(arr))
    with _reference_cache_lock:
        if _reference_cache.get("key") != key:
            wav = _trim_silence(_prepare_waveform(arr, reference_sr))
            _reference_cache = {
                "key": key,
                "wav": wav,
                "embeddings": extract_embeddings(wav),
            }
        return _reference_cache

# =====================================================================
# 唯一入口
# =====================================================================
# 样本日志上限：每次进程里、在第一次追加之前，把日志裁到最新的
# MAX_SAMPLES_KEPT 行，避免无限变长。比 calibrate.py 的 MAX_SAMPLES_USED（300）
# 宽裕，裁剪不会饿死校准。与音素引擎的裁剪（pronunciation/phoneme/speech.py）对称。
MAX_SAMPLES_KEPT = 2000

_samples_trimmed = False  # 本进程只让 _trim_sample_log() 跑一次的开关

def _trim_sample_log() -> None:
    """样本日志只留下最新的 MAX_SAMPLES_KEPT 行。

    由 _append_calibration_sample 每个进程调用一次，这样稳态追加仍是 O(1) 写入，
    文件只在两次运行之间变短。
    """
    path = samples_file()
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= MAX_SAMPLES_KEPT:
        return
    # lines[-N:]：切片，从倒数第 N 行一直到末尾（最新的那些行）。
    path.write_text("\n".join(lines[-MAX_SAMPLES_KEPT:]) + "\n", encoding="utf-8")
    logging.info("Sample log trimmed: %d -> %d lines (%s)",
                 len(lines), MAX_SAMPLES_KEPT, path.name)

def _append_calibration_sample(record: dict[str, Any]) -> None:
    """把一条分析记录追加到校准样本日志（尽力而为）。

    这文件喂给 ``acoustic/calibrate.py``；写失败绝不能把分析本身弄崩。
    """
    global _samples_trimmed
    try:
        if not _samples_trimmed:
            _samples_trimmed = True  # 先置位：裁剪失败也不能每录一次就重试
            _trim_sample_log()
        path = samples_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        # with 打开文件：离开块时自动关闭。模式 "a" 是追加。
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logging.exception("Failed to append calibration sample:")

def analyze(user_audio: np.ndarray,
            expected_text: str,
            reference_audio: np.ndarray | None = None,
            user_sr: int = TARGET_SAMPLE_RATE,
            reference_sr: int = KOKORO_SAMPLE_RATE,
            voice: str | None = None,
            is_reference: bool = False) -> PronunciationResult:
    """把用户的开口尝试和期望短语做对比。

    功能：本引擎的唯一入口。抽嵌入、做 DTW 声学分，再转写并对比音素 / 字符，
        返回统一的 PronunciationResult。没有参考音频就无法打声学分为此会报错。

    参数：
        user_audio: 用户录音波形（一维 float32；来自录音路径）。
        expected_text: 用户被要求跟读的参考短语。
        reference_audio: 同一短语的 Kokoro 合成参考波形。标成可选只是为了
            和音素 / none 引擎的签名对齐（「对外 API 与 acoustic/ 完全镜像」）：
            本引擎没有它就打不了分——声学分量就是在和参考比——所以传入 None
            会抛 ValueError，把问题说清楚；若做成仅位置参数，那边只会抛 TypeError。
            ``np.ndarray | None`` 表示数组或 None。
        user_sr: ``user_audio`` 的采样率（录音路径是 16 kHz）。
        reference_sr: ``reference_audio`` 的采样率（Kokoro 是 24 kHz）。
        voice: 合成参考时用的 Kokoro 音色。只记进校准样本日志——声学下限和音色有关，
            calibrate.py 需要知道每条样本是哪条音色出的。
        is_reference: 标记「参考音自己和自己比」的自测。接这个参数是为了让分发器
            能用同一套签名调每个引擎；本引擎忽略它——它的 calibrate.py 已经靠
            接近零的声学距离（SELF_TEST_ACOUSTIC）排除自测，这里不需要旗标。

    返回：
        PronunciationResult，含分数、按词错误、韵律占位和转写。
    """
    if reference_audio is None:
        raise ValueError(
            "the acoustic engine requires reference_audio: its score is a "
            "comparison against the reference recording")
    _ensure_loaded()

    # 剪掉两端静音填充：用户录音有按键停顿（峰值归一化还会把底噪抬高），
    # TTS 参考几乎没有。
    user_wav = _trim_silence(_prepare_waveform(user_audio, user_sr))
    reference = _reference_features(reference_audio, reference_sr)

    # 声学相似度：两条嵌入序列的余弦 DTW，再除以对齐路径长度，
    # 这样不会随句子变长而变大。用余弦（而不是欧氏）还会忽略嵌入的模长，
    # 模长会随响度 / 音色漂，而不是随发音漂。
    emb_user = extract_embeddings(user_wav)
    emb_reference = reference["embeddings"]
    acoustic_total, path = fastdtw(emb_user, emb_reference, dist=cosine)
    acoustic_per_step = float(acoustic_total) / max(1, len(path))
    acoustic_baseline = _random_pair_baseline(emb_user, emb_reference)
    acoustic_bad = acoustic_bad_for(acoustic_baseline)

    # 转写 + 按词的音素对比。
    transcription = clean_transcription(transcribe(user_wav))
    differences = compare_transcriptions(transcription, expected_text)

    phoneme_length = max(1, differences["phoneme_length"])
    phoneme_error_rate = differences["phoneme_distance"] / phoneme_length
    reference_length = max(1, differences["reference_length"])
    # 这个比率仍用历史名字 "word_error_rate"：它属于打分 API，
    # 也是 calibrate.py 要读的校准日志字段。
    word_error_rate = differences["char_distance"] / reference_length

    score = compute_pronunciation_score(
        acoustic_per_step,
        phoneme_error_rate,
        word_error_rate,
        acoustic_bad=acoustic_bad,
    )

    # 校准日志：原始分量写在一行里方便 grep。``calibrate.py`` 吃的是
    # 下面追加到样本文件里的那份结构化拷贝。
    logging.info(
        "[acoustic] score=%.1f | acoustic/step=%.4f (good=%.3f bad=%.3f baseline=%.4f) | "
        "phonemes=%d/%d (err=%.2f) | chars_lev=%d/%d (err=%.2f) | voice=%s | asr=%r",
        score, acoustic_per_step, current_acoustic_floor(), acoustic_bad, acoustic_baseline,
        differences["phoneme_distance"], phoneme_length, phoneme_error_rate,
        differences["char_distance"], reference_length, word_error_rate,
        voice, transcription,
    )
    _append_calibration_sample({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "text": expected_text,
        "asr": transcription,
        # 练习用户（AnalyzerConfig.user_name，未设置时为 ""）。声学下限按用户分，
        # 所以 calibrate.py 用这个字段过滤日志。
        "user_name": get_config().user_name,
        "voice": voice,
        "acoustic_per_step": round(acoustic_per_step, 5),
        "acoustic_baseline": round(acoustic_baseline, 5),
        "phoneme_distance": int(differences["phoneme_distance"]),
        "phoneme_length": int(phoneme_length),
        # 从 "word_distance" 改名（其实是字符级）；calibrate.py 只读
        # *_error_rate 字段，所以旧样本行仍然能用。
        "char_distance": int(differences["char_distance"]),
        "reference_length": int(reference_length),
        "phoneme_error_rate": round(phoneme_error_rate, 4),
        "word_error_rate": round(word_error_rate, 4),
        "score": score,
    })

    # 韵律是引擎无关的音频层（mimora/prosody.py），由宿主从原始波形计算。
    # 引擎返回空字典；宿主填好后再把结果交给 UI。
    return PronunciationResult(
        score=score,
        word_errors=differences["errors"],
        prosody={},
        transcription=transcription,
        passed=score >= get_config().score_threshold,
        feedback=differences["feedback"],
        acoustic_distance=int(acoustic_total),
        acoustic_per_step=acoustic_per_step,
        acoustic_baseline=acoustic_baseline,
        words_with_errors=differences["words_with_errors"],
        expected_phonemes=differences["expected_phonemes"],
        transcribed_phonemes=differences["transcribed_phonemes"],
        word_diff=word_level_diff(transcription, expected_text),
        # 引擎无关的展示字段。声学引擎的「单位」是词：
        # recognized_units 镜像 heard_word_tags，只是键改成中性的 "unit"。
        reference_words=reference_word_tags(expected_text, differences["words_with_errors"]),
        recognized_units=[{"unit": t["word"], "correct": t["correct"]}
                          for t in heard_word_tags(transcription, expected_text)],
        # 声学引擎没有按音素的细分，排不出薄弱音素；显式传入空列表，
        # 好让结果 API 和音素引擎一致。GUI 在这里会退回它的 "Heard" 行。
        weak_phonemes=[],
    )
