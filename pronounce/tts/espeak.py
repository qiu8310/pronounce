"""Isolated-phone TTS via espeak-ng (Kokoro cannot speak IPA)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ESPEAK_SAMPLE_RATE = 22_050
_PAD_SEC = 0.15
_RATE = 100

# Unicode IPA inside [[ ]] is silent. These are espeak-ng phoneme mnemonics.
_IPA_TO_MNEM = {
    "ɪ": "I",
    "ɛ": "E",
    "æ": "a",
    "ɒ": "0",
    "ʌ": "V",
    "ʊ": "U",
    "ɐ": "@",
    "ə": "@",
    "iː": "i:",
    "ɜː": "3:",
    "ɑː": "A:",
    "ɔː": "O:",
    "uː": "u:",
    "eɪ": "eI",
    "aɪ": "aI",
    "ɔɪ": "OI",
    "əʊ": "@U",
    "aʊ": "aU",
    "iə": "I@",
    "ɪə": "I@",
    "eə": "e@",
    "ʊə": "U@",
    "ɡ": "g@",
    "g": "g@",
    "θ": "T",
    "ð": "D",
    "ʃ": "S",
    "ʒ": "Z",
    "ŋ": "N",
    "tʃ": "tS",
    "dʒ": "dZ@",
    "ɹ": "r",
    "tɹ": "tr",
    "dɹ": "dr",
    "b": "b@",
    "d": "d@",
}


def espeak_phonemes(ipa: str) -> str:
    phone = (ipa or "").strip()
    return _IPA_TO_MNEM.get(phone, phone)


def espeak_bin() -> str:
    path = shutil.which("espeak-ng") or shutil.which("espeak")
    if not path:
        raise FileNotFoundError("espeak-ng is not on PATH")
    return path


def synthesize_ipa(ipa: str, *, lang: str = "en-gb") -> tuple[np.ndarray, int]:
    """Speak one IPA phone. Returns mono float32 and sample rate."""
    phone = espeak_phonemes(ipa)
    if not phone:
        raise ValueError("empty ipa")
    if len(phone) > 16 or "[" in phone or "]" in phone:
        raise ValueError(f"invalid ipa: {ipa!r}")

    voice = "en-gb" if lang.lower() in {"en-gb", "en-uk", "b"} else "en-us"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        subprocess.run(
            [espeak_bin(), "-v", voice, "-s", str(_RATE), "-w", str(out), f"[[{phone}]]"],
            check=True,
            capture_output=True,
        )
        import soundfile as sf

        audio, sr = sf.read(str(out), dtype="float32", always_2d=False)
    finally:
        out.unlink(missing_ok=True)

    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    pad = int(sr * _PAD_SEC)
    if pad > 0:
        wave = np.pad(wave, (pad, pad))
    return wave, int(sr)
