from __future__ import annotations

from pronounce.common import PronunciationResult


def to_payload(
    *,
    engine: str,
    result: PronunciationResult,
    text: str,
    user_wav: str,
    ref_wav: str | None,
) -> dict:
    phoneme: dict
    acoustic: dict
    if engine == "phoneme":
        phoneme = {
            "bucket": result.bucket,
            "user_percent": result.user_percent,
            "grade": result.grade,
            "grade_value": result.grade_value,
            "expected_phonemes": result.expected_phonemes,
            "transcribed_phonemes": result.transcribed_phonemes,
            "weak_phonemes": result.weak_phonemes,
            "ipa_words": result.ipa_words,
            "per_phone_distance": result.per_phone_distance,
            "bad_baseline": result.bad_baseline,
            "phoneme_score": result.phoneme_score,
            "recall": result.recall,
            "good_anchor": result.good_anchor,
        }
        acoustic = {}
    elif engine == "acoustic":
        phoneme = {}
        acoustic = {
            "acoustic_distance": result.acoustic_distance,
            "acoustic_per_step": result.acoustic_per_step,
            "acoustic_baseline": result.acoustic_baseline,
        }
    else:
        phoneme = {}
        acoustic = {}

    return {
        "ok": True,
        "engine": engine,
        "text": text,
        "user_wav": user_wav,
        "ref_wav": ref_wav,
        "score": result.score,
        "passed": result.passed,
        "scored": result.scored,
        "transcription": result.transcription,
        "feedback": result.feedback,
        "word_errors": result.word_errors,
        "words_with_errors": result.words_with_errors,
        "word_diff": result.word_diff,
        "reference_words": result.reference_words,
        "recognized_units": result.recognized_units,
        "prosody": {},
        "phoneme": phoneme,
        "acoustic": acoustic,
    }
