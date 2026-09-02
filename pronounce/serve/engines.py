from __future__ import annotations

import threading

from pronounce.paths import wav2vec2_model
from pronounce.phonemes.ipa import ipa_for_text
from pronounce.tts import to_file as tts_to_file
from pronounce.tts.kokoro import load_tts

# Phoneme AnalyzerConfig is process-global; serialize configure+analyze under ThreadingHTTPServer.
_score_lock = threading.Lock()

__all__ = ["warmup", "tts_to_file", "dictionary_ipa", "score_phoneme"]


def warmup(device: str = "cpu") -> None:
    from pronounce.score.phoneme import AnalyzerConfig, configure, load_models

    with _score_lock:
        configure(
            AnalyzerConfig(
                model_name=str(wav2vec2_model("wav2vec2-xlsr-53-espeak-cv-ft")),
                device=device,
                espeak_language="en-us",
            )
        )
        load_models()
    load_tts(device)


def dictionary_ipa(*, text: str, lang: str = "en-us") -> dict:
    payload = ipa_for_text(text, lang=lang)
    return {"ok": True, "command": "phonemes", "text": text, "lang": lang, **payload}


def score_phoneme(
    *,
    text: str | None,
    user_wav: str,
    ref_wav: str | None = None,
    lang: str = "en-us",
    device: str = "cpu",
    ipa: str | None = None,
) -> dict:
    from pronounce.score.jobs import score_phoneme as _score_phoneme

    with _score_lock:
        return _score_phoneme(
            text=text,
            user_wav=user_wav,
            ref_wav=ref_wav,
            lang=lang,
            device=device,
            ipa=ipa,
        )
