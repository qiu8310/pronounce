"""Play PCM in-process via PortAudio. No files."""

from __future__ import annotations

import numpy as np


def require_destination(out: str | None, play: bool) -> None:
    if not (out or "").strip() and not play:
        raise ValueError("out or play is required")


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    import sounddevice as sd

    sd.play(np.asarray(audio), samplerate=int(sample_rate))
    sd.wait()
