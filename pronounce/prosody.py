"""Pitch and energy contours from user vs reference waveforms.

Copied from mimora's host-side layer: engines leave ``prosody`` empty; the CLI
fills it so a UI can draw intonation. No torch.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import librosa
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

_reference_cache: Dict[str, Any] = {}


def extract_f0(audio_waveform: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    f0, _voiced_flag, _voiced_probs = librosa.pyin(
        audio_waveform, fmin=50, fmax=450, sr=sr
    )
    return np.nan_to_num(f0)


def extract_energy(audio_waveform: np.ndarray) -> np.ndarray:
    energy = librosa.feature.rms(y=audio_waveform)
    scaler = MinMaxScaler(feature_range=(0, 250))
    return scaler.fit_transform(energy.T).flatten()


def interpolate_f0(f0: np.ndarray) -> np.ndarray:
    f0 = np.array(f0)
    mask = f0 > 0
    if not mask.any():
        return f0
    return np.interp(np.arange(len(f0)), np.where(mask)[0], f0[mask])


def _reference_prosody(reference_audio: np.ndarray, reference_sr: int) -> Dict[str, np.ndarray]:
    global _reference_cache
    arr = np.asarray(reference_audio)
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
) -> Dict[str, List[float]]:
    """Four contours: ``f0``, ``energy``, ``ref_f0``, ``ref_energy``."""
    user_wav = _trim_silence(_prepare_waveform(user_audio, user_sr))
    reference = _reference_prosody(reference_audio, reference_sr)
    return {
        "f0": interpolate_f0(extract_f0(user_wav, TARGET_SAMPLE_RATE)).tolist(),
        "energy": extract_energy(user_wav).tolist(),
        "ref_f0": reference["f0"].tolist(),
        "ref_energy": reference["energy"].tolist(),
    }


def user_only_prosody(user_audio: np.ndarray, user_sr: int) -> Dict[str, List[float]]:
    """When there is no reference wav: user contours, empty ref lists."""
    user_wav = _trim_silence(_prepare_waveform(user_audio, user_sr))
    return {
        "f0": interpolate_f0(extract_f0(user_wav, TARGET_SAMPLE_RATE)).tolist(),
        "energy": extract_energy(user_wav).tolist(),
        "ref_f0": [],
        "ref_energy": [],
    }
