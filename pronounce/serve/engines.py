from __future__ import annotations

import threading
from pathlib import Path

from pronounce.paths import wav2vec2_model
from pronounce.phonemes.ipa import ipa_for_text
from pronounce.tts import DEFAULT_VOICE, KOKORO_SAMPLE_RATE, synthesize
from pronounce.tts.kokoro import load_tts

# Phoneme AnalyzerConfig is process-global; serialize configure+analyze under ThreadingHTTPServer.
_score_lock = threading.Lock()


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


def tts_to_file(*, text: str, out: str, voice: str = DEFAULT_VOICE, lang: str = "en-us") -> dict:
    import soundfile as sf

    path = Path(out).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = synthesize(text, voice=voice, lang=lang, device="cpu")
    sf.write(str(path), audio, KOKORO_SAMPLE_RATE)
    return {
        "ok": True,
        "command": "tts",
        "text": text,
        "voice": voice,
        "lang": lang,
        "out": str(path),
        "speed": 1.0,
        "sample_rate": KOKORO_SAMPLE_RATE,
        "native_rate": KOKORO_SAMPLE_RATE,
    }


def dictionary_ipa(*, text: str, lang: str = "en-us") -> dict:
    payload = ipa_for_text(text, lang=lang)
    return {"ok": True, "command": "phonemes", "text": text, "lang": lang, **payload}


def score_phoneme(*, text: str, user_wav: str, ref_wav: str, lang: str = "en-us", device: str = "cpu") -> dict:
    from pronounce.common.audio import TARGET_SAMPLE_RATE, prepare_waveform
    from pronounce.score.json_out import to_payload
    from pronounce.score.phoneme import AnalyzerConfig, analyze, configure, load_models
    from pronounce.score.prosody import compute_prosody

    user_path = Path(user_wav).resolve()
    ref_path = Path(ref_wav).resolve()
    if not user_path.is_file():
        raise FileNotFoundError(f"user audio not found: {user_path}")
    if not ref_path.is_file():
        raise FileNotFoundError(f"reference audio not found: {ref_path}")

    import soundfile as sf

    user_audio, user_sr = sf.read(str(user_path), dtype="float32", always_2d=False)
    ref_audio, ref_sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
    user_audio = prepare_waveform(user_audio, int(user_sr))
    ref_audio = prepare_waveform(ref_audio, int(ref_sr))

    with _score_lock:
        configure(
            AnalyzerConfig(
                model_name=str(wav2vec2_model("wav2vec2-xlsr-53-espeak-cv-ft")),
                device=device,
                espeak_language=lang,
            )
        )
        load_models()
        result = analyze(
            user_audio,
            text,
            reference_audio=ref_audio,
            user_sr=TARGET_SAMPLE_RATE,
            reference_sr=TARGET_SAMPLE_RATE,
        )
        contours = compute_prosody(user_audio, TARGET_SAMPLE_RATE, ref_audio, TARGET_SAMPLE_RATE)
        return to_payload(
            engine="phoneme",
            result=result,
            text=text,
            user_wav=str(user_path),
            ref_wav=str(ref_path),
            prosody=contours,
        )
