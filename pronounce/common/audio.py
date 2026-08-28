# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""发音栈共用的、不依赖 torch 的波形处理。

声学 / 音素两个引擎以及宿主的语调层（mimora/prosody.py）必须用同一套预处理：
分数和语调曲线要测在同一段信号上。如果各写一份，时间一长就会漂移。
本模块是唯一副本，放在 ``pronounce.common`` 里，大家都允许依赖。

刻意不用 torch/transformers；librosa 在函数内部才 import（惰性导入），
这样 ``import pronounce.common.audio`` 不会把整套 ML 栈拉进来。
"""

from __future__ import annotations

import hashlib

import numpy as np

# Wav2Vec2 识别器和语调分析都要求严格 16 kHz 单声道。
TARGET_SAMPLE_RATE = 16_000

# 相对峰值的静音裁剪阈值（dB）。三处共用，裁完的信号才能对得上。
TRIM_TOP_DB = 30


def prepare_waveform(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
    """把任意布局的波形变成 1 维 float32 单声道，并重采样到 TARGET_SAMPLE_RATE。"""
    import librosa  # 惰性导入：调用时才加载，模块 import 保持轻量

    wav = np.asarray(waveform, dtype=np.float32)

    # 混成单声道。加载器布局不统一：soundfile 是 [采样, 声道]，
    # torch 系常是 [声道, 采样]，所以沿更短的那一轴（声道轴）求平均。
    if wav.ndim > 1:
        wav = wav.mean(axis=int(np.argmin(wav.shape)))

    if orig_sr != TARGET_SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)

    # ascontiguousarray：保证内存连续（C 顺序），后续传给 C 扩展 / 哈希时更稳。
    return np.ascontiguousarray(wav, dtype=np.float32)


def trim_silence(wav: np.ndarray) -> np.ndarray:
    """裁掉首尾静音，避免空白把分数和语调曲线拉歪。

    用户录音尤其需要：采集端峰值归一化会抬高安静片段的底噪，
    静音填充会变成「很响的噪声」，干净的 TTS 参考里没有对应物。
    若裁完不足 0.1 秒（几乎全是静音）则原样返回。
    """
    import librosa

    if wav.size == 0:
        return wav
    trimmed, _ = librosa.effects.trim(wav, top_db=TRIM_TOP_DB)
    if trimmed.size < int(0.1 * TARGET_SAMPLE_RATE):
        return wav
    return np.ascontiguousarray(trimmed, dtype=np.float32)


def waveform_digest(waveform: np.ndarray) -> bytes:
    """波形内容的稳定摘要，用来当缓存键。

    参考音频的嵌入 / 识别音素 / 语调曲线都要按「同一段波形」复用。
    用 memoryview 再 .cast("B") 按字节视图哈希，避免 arr.tobytes() 再拷一份整段音频。
    SHA-1 是真正的内容摘要；Python 内置 hash() 只有 64 位且进程间不稳定。
    """
    arr = np.ascontiguousarray(waveform)
    # memoryview(...).cast("B")：把数组看成无类型字节序列，不复制数据。
    return hashlib.sha1(memoryview(arr).cast("B")).digest()
