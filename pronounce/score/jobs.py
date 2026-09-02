"""音素打分的文件入口：CLI ``score phoneme`` 和 ``pronounce serve`` 共用。"""

from __future__ import annotations

from pathlib import Path

from pronounce.paths import wav2vec2_model


def score_phoneme(
    *,
    text: str | None,
    user_wav: str,
    ref_wav: str | None = None,
    lang: str = "en-us",
    device: str = "cpu",
    ipa: str | None = None,
    calibration: str | Path | None = None,
    user_name: str = "",
) -> dict:
    """读 wav、跑 phoneme ``analyze``，返回与 CLI / HTTP ``score`` 相同的成功 JSON。"""
    from pronounce.common.audio import TARGET_SAMPLE_RATE, prepare_waveform
    from pronounce.score.json_out import to_payload
    from pronounce.score.phoneme import AnalyzerConfig, analyze, configure, load_models
    from pronounce.score.prosody import compute_prosody, user_only_prosody

    ipa = (ipa or "").strip() or None
    text = (text or "").strip() or (f"/{ipa}/" if ipa else "")
    if not text:
        raise ValueError("text or ipa is required")

    user_path = Path(user_wav).expanduser().resolve()
    ref_path = Path(ref_wav).expanduser().resolve() if ref_wav else None
    if not user_path.is_file():
        raise FileNotFoundError(f"user audio not found: {user_path}")
    if ref_path is not None and not ref_path.is_file():
        raise FileNotFoundError(f"reference audio not found: {ref_path}")

    import soundfile as sf

    user_audio, user_sr = sf.read(str(user_path), dtype="float32", always_2d=False)
    user_audio = prepare_waveform(user_audio, int(user_sr))
    reference_audio = None
    if ref_path is not None:
        reference_audio, ref_sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
        reference_audio = prepare_waveform(reference_audio, int(ref_sr))

    cal = Path(calibration).expanduser().resolve() if calibration else None
    configure(
        AnalyzerConfig(
            model_name=str(wav2vec2_model("wav2vec2-xlsr-53-espeak-cv-ft")),
            device=device,
            espeak_language=lang,
            user_name=user_name or "",
            calibration_file=cal,
        )
    )
    load_models()
    result = analyze(
        user_audio,
        text,
        reference_audio=reference_audio,
        user_sr=TARGET_SAMPLE_RATE,
        reference_sr=TARGET_SAMPLE_RATE,
        expected_ipa=ipa,
    )
    if reference_audio is not None:
        contours = compute_prosody(
            user_audio, TARGET_SAMPLE_RATE, reference_audio, TARGET_SAMPLE_RATE
        )
    else:
        contours = user_only_prosody(user_audio, TARGET_SAMPLE_RATE)
    return to_payload(
        engine="phoneme",
        result=result,
        text=text,
        user_wav=str(user_path),
        ref_wav=str(ref_path) if ref_path is not None else None,
        prosody=contours,
    )
