from __future__ import annotations

import threading

from pronounce.paths import wav2vec2_model
from pronounce.phonemes.ipa import ipa_for_text
from pronounce.phonemes.style import normalize_style
from pronounce.tts import to_file as tts_to_file
from pronounce.tts.kokoro import load_tts

_phoneme_lock = threading.Lock()
_acoustic_lock = threading.Lock()

__all__ = [
    "warmup",
    "tts_to_file",
    "dictionary_ipa",
    "score_phoneme",
    "score_acoustic",
]


def warmup(device: str = "cpu") -> None:
    from pronounce.score.acoustic import (
        AnalyzerConfig as AcousticConfig,
        configure as configure_acoustic,
        load_models as load_acoustic,
    )
    from pronounce.score.phoneme import (
        AnalyzerConfig as PhonemeConfig,
        configure as configure_phoneme,
        load_models as load_phoneme,
    )

    with _phoneme_lock:
        configure_phoneme(
            PhonemeConfig(
                model_name=str(wav2vec2_model("wav2vec2-xlsr-53-espeak-cv-ft")),
                device=device,
                espeak_language="en-us",
            )
        )
        load_phoneme()
    with _acoustic_lock:
        configure_acoustic(
            AcousticConfig(
                model_name=str(wav2vec2_model("wav2vec2-large-960h")),
                device=device,
                espeak_language="en-us",
            )
        )
        load_acoustic()
    load_tts(device)


def dictionary_ipa(*, text: str, lang: str = "en-us", style: str | None = None) -> dict:
    st = normalize_style(style)
    payload = ipa_for_text(text, lang=lang, style=st)
    return {
        "ok": True,
        "command": "phonemes",
        "text": text,
        "lang": lang,
        "style": st,
        **payload,
    }


def score_phoneme(
    *,
    text: str | None,
    user_wav: str,
    ref_wav: str | None = None,
    lang: str = "en-us",
    device: str = "cpu",
    ipa: str | None = None,
    style: str | None = None,
) -> dict:
    from pronounce.score.jobs import score_phoneme as _score_phoneme

    with _phoneme_lock:
        return _score_phoneme(
            text=text,
            user_wav=user_wav,
            ref_wav=ref_wav,
            lang=lang,
            device=device,
            ipa=ipa,
            style=style,
        )


def score_acoustic(
    *,
    text: str,
    user_wav: str,
    ref_wav: str,
    lang: str = "en-us",
    device: str = "cpu",
) -> dict:
    from pronounce.score.jobs import score_acoustic as _score_acoustic

    with _acoustic_lock:
        return _score_acoustic(
            text=text,
            user_wav=user_wav,
            ref_wav=ref_wav,
            lang=lang,
            device=device,
        )
