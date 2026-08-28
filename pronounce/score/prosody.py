"""从用户 / 参考波形抽出音高（F0）和能量曲线。

从 mimora 宿主侧拷过来：引擎把 ``prosody`` 留空；CLI 在这里填，UI 才能画语调。
不依赖 torch。
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pronounce.common.audio import (
    TARGET_SAMPLE_RATE,
    prepare_waveform as _prepare_waveform,
    trim_silence as _trim_silence,
    waveform_digest,
)

__all__ = [
    "TARGET_SAMPLE_RATE",
    "compute_prosody",
    "extract_energy",
    "extract_f0",
    "interpolate_f0",
    "user_only_prosody",
]

# 参考音的曲线缓存：同一段参考被多次打分时不必重算 pyin / RMS。
_reference_cache: dict[str, Any] = {}


def extract_f0(audio_waveform: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """用 librosa.pyin 估基频；无声帧是 NaN，这里换成 0。"""
    f0, _voiced_flag, _voiced_probs = librosa.pyin(
        audio_waveform, fmin=50, fmax=450, sr=sr
    )
    return np.nan_to_num(f0)


def extract_energy(audio_waveform: np.ndarray) -> np.ndarray:
    """逐帧 RMS 能量，再线性缩放到 0–250，方便和 F0 画在同一张图上。"""
    energy = librosa.feature.rms(y=audio_waveform)
    scaler = MinMaxScaler(feature_range=(0, 250))
    # energy 形状是 (1, 帧数)；.T 变成 (帧数, 1) 才能交给 scaler。
    return scaler.fit_transform(energy.T).flatten()


def interpolate_f0(f0: np.ndarray) -> np.ndarray:
    """把无声（0）帧用两侧有声帧线性插值填上，曲线才连续。

    np.interp(x, xp, fp)：在 xp 的采样点上已知 fp，求 x 处的值。
    """
    f0 = np.array(f0)
    mask = f0 > 0
    if not mask.any():
        return f0
    return np.interp(np.arange(len(f0)), np.where(mask)[0], f0[mask])


def _reference_prosody(reference_audio: np.ndarray, reference_sr: int) -> dict[str, np.ndarray]:
    """算参考音的 F0 / energy；按 (采样率, 形状, 内容摘要) 缓存。"""
    global _reference_cache
    arr = np.asarray(reference_audio)
    # 元组可哈希，适合当缓存键；digest 避免只比 shape 导致「两段不同音频撞车」。
    key = (reference_sr, arr.shape, waveform_digest(arr))
    if _reference_cache.get("key") != key:
        wav = _trim_silence(_prepare_waveform(arr, reference_sr))
        _reference_cache = {
            "key": key,
            "f0": interpolate_f0(extract_f0(wav, TARGET_SAMPLE_RATE)),
            "energy": extract_energy(wav),
        }
    return _reference_cache


def compute_prosody(
    user_audio: np.ndarray,
    user_sr: int,
    reference_audio: np.ndarray,
    reference_sr: int,
) -> dict[str, list[float]]:
    """四条曲线：``f0``、``energy``、``ref_f0``、``ref_energy``（都是 Python list，方便 JSON）。"""
    user_wav = _trim_silence(_prepare_waveform(user_audio, user_sr))
    reference = _reference_prosody(reference_audio, reference_sr)
    return {
        "f0": interpolate_f0(extract_f0(user_wav, TARGET_SAMPLE_RATE)).tolist(),
        "energy": extract_energy(user_wav).tolist(),
        "ref_f0": reference["f0"].tolist(),
        "ref_energy": reference["energy"].tolist(),
    }


def user_only_prosody(user_audio: np.ndarray, user_sr: int) -> dict[str, list[float]]:
    """没有参考 wav 时：只填用户曲线，参考两条留空列表。"""
    user_wav = _trim_silence(_prepare_waveform(user_audio, user_sr))
    return {
        "f0": interpolate_f0(extract_f0(user_wav, TARGET_SAMPLE_RATE)).tolist(),
        "energy": extract_energy(user_wav).tolist(),
        "ref_f0": [],
        "ref_energy": [],
    }
