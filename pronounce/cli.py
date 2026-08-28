from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from pronounce.paths import wav2vec2_model

_FIELDS_MD = Path(__file__).resolve().parents[1] / "FIELDS.md"


def _fail(engine: str, error: str) -> int:
    print(json.dumps({"ok": False, "engine": engine, "error": error}))
    return 1


def _check_model_dir(engine: str, model_dir: str) -> str | None:
    root = Path(model_dir)
    if not root.is_dir() or not (root / "config.json").is_file():
        return (
            f"model directory missing or incomplete at {model_dir}; "
            f"expected a from_pretrained root under llm/wav2vec2/"
        )
    return None


def _load_wav(path: str):
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    return data, int(sr)


def _fail_plain(error: str) -> int:
    print(json.dumps({"ok": False, "error": error}))
    return 1


def _cmd_schema(_args: argparse.Namespace) -> int:
    text = _FIELDS_MD.read_text(encoding="utf-8")
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def _cmd_tts(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    try:
        from pronounce.kokoro import KOKORO_SAMPLE_RATE, listen_sample_rate, synthesize

        rate = listen_sample_rate(args.speed, KOKORO_SAMPLE_RATE)
        audio = synthesize(
            args.text, voice=args.voice, lang=args.lang, device=args.device
        )
        import soundfile as sf

        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, rate)
        print(
            json.dumps(
                {
                    "ok": True,
                    "command": "tts",
                    "text": args.text,
                    "voice": args.voice,
                    "lang": args.lang,
                    "out": str(out),
                    "speed": args.speed,
                    "sample_rate": rate,
                    "native_rate": KOKORO_SAMPLE_RATE,
                }
            )
        )
        return 0
    except Exception as e:
        return _fail_plain(str(e))


def _cmd_phonemes(args: argparse.Namespace) -> int:
    try:
        from pronounce.ipa import ipa_for_text

        payload = ipa_for_text(args.text, lang=args.lang)
        print(
            json.dumps(
                {
                    "ok": True,
                    "command": "phonemes",
                    "text": args.text,
                    "lang": args.lang,
                    **payload,
                }
            )
        )
        return 0
    except Exception as e:
        return _fail_plain(str(e))


def _synthesize_ref(args: argparse.Namespace) -> Path:
    """Write a native-speed Kokoro wav and return its path."""
    from pronounce.kokoro import KOKORO_SAMPLE_RATE, synthesize
    import soundfile as sf

    audio = synthesize(
        args.text, voice=args.voice, lang=args.lang, device=args.device
    )
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="pronounce-ref-")
    os.close(fd)
    path = Path(tmp)
    sf.write(str(path), audio, KOKORO_SAMPLE_RATE)
    return path


def _cmd_score(args: argparse.Namespace) -> int:
    engine = args.engine
    model_name = str(
        wav2vec2_model(
            "wav2vec2-xlsr-53-espeak-cv-ft"
            if engine == "phoneme"
            else "wav2vec2-large-960h"
        )
    )
    model_err = _check_model_dir(engine, model_name)
    if model_err is not None:
        return _fail(engine, model_err)

    user_path = Path(args.user).resolve()
    ref_path = Path(args.ref).resolve() if args.ref else None
    ref_generated = False

    if not user_path.is_file():
        return _fail(engine, f"user audio not found: {user_path}")
    if ref_path is not None and not ref_path.is_file():
        return _fail(engine, f"reference audio not found: {ref_path}")

    try:
        from pronounce.common.audio import TARGET_SAMPLE_RATE, prepare_waveform
        from pronounce.json_out import to_payload
        from pronounce.prosody import compute_prosody, user_only_prosody

        if engine == "acoustic" and ref_path is None:
            ref_path = _synthesize_ref(args)
            ref_generated = True

        user_audio, user_sr = _load_wav(str(user_path))
        user_audio = prepare_waveform(user_audio, user_sr)
        user_sr = TARGET_SAMPLE_RATE

        reference_audio = None
        reference_sr = TARGET_SAMPLE_RATE
        if ref_path is not None:
            reference_audio, reference_sr = _load_wav(str(ref_path))
            reference_audio = prepare_waveform(reference_audio, reference_sr)
            reference_sr = TARGET_SAMPLE_RATE

        if engine == "phoneme":
            from pronounce.phoneme import AnalyzerConfig, analyze, configure, load_models
        else:
            from pronounce.acoustic import AnalyzerConfig, analyze, configure, load_models

        cal = Path(args.calibration).expanduser().resolve() if args.calibration else None
        configure(
            AnalyzerConfig(
                model_name=model_name,
                device=args.device,
                espeak_language=args.lang,
                user_name=args.user_name or "",
                calibration_file=cal,
            )
        )
        load_models()
        result = analyze(
            user_audio,
            args.text,
            reference_audio=reference_audio,
            user_sr=user_sr,
            reference_sr=reference_sr,
        )
        if reference_audio is not None:
            contours = compute_prosody(
                user_audio, user_sr, reference_audio, reference_sr
            )
        else:
            contours = user_only_prosody(user_audio, user_sr)
        payload = to_payload(
            engine=engine,
            result=result,
            text=args.text,
            user_wav=str(user_path),
            ref_wav=str(ref_path) if ref_path is not None else None,
            prosody=contours,
        )
        if ref_generated:
            payload["ref_generated"] = True
        print(json.dumps(payload))
        return 0
    except Exception as e:
        return _fail(engine, str(e))


def _engine_from_argv(argv: list[str]) -> str | None:
    """Return phoneme/acoustic when already present in argv, else None."""
    try:
        score_idx = argv.index("score")
    except ValueError:
        return None
    if score_idx + 1 >= len(argv):
        return None
    engine = argv[score_idx + 1]
    if engine in ("phoneme", "acoustic"):
        return engine
    return None


class _ArgError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class _JsonArgumentParser(argparse.ArgumentParser):
    """Emit stdout JSON failures instead of argparse usage text / exit 2."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise _ArgError(message)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _JsonArgumentParser(prog="pronounce")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score a spoken take")
    score.add_argument("engine", choices=("phoneme", "acoustic"))
    score.add_argument("--text", required=True)
    score.add_argument("--user", required=True)
    score.add_argument("--ref", default=None)
    score.add_argument("--lang", default="en-us")
    score.add_argument("--device", default="cpu")
    score.add_argument("--voice", default="af_heart", help="Kokoro voice when --ref is omitted (acoustic)")
    score.add_argument("--calibration", default=None, help="per-user calibration.json")
    score.add_argument("--user-name", default="", dest="user_name")
    score.set_defaults(func=_cmd_score)

    schema = sub.add_parser("schema", help="print FIELDS.md")
    schema.set_defaults(func=_cmd_schema)

    tts = sub.add_parser("tts", help="synthesize English speech with Kokoro")
    tts.add_argument("--text", required=True)
    tts.add_argument("--out", required=True)
    tts.add_argument("--voice", default="af_heart")
    tts.add_argument("--lang", default="en-us")
    tts.add_argument("--device", default="cpu")
    tts.add_argument("--speed", type=float, default=1.0, help="playback tempo; 0.8 is slower (lower wav sample rate)")
    tts.set_defaults(func=_cmd_tts)

    phonemes = sub.add_parser("phonemes", help="dictionary IPA for people to read")
    phonemes.add_argument("--text", required=True)
    phonemes.add_argument("--lang", default="en-us")
    phonemes.set_defaults(func=_cmd_phonemes)

    try:
        args = parser.parse_args(argv)
    except _ArgError as e:
        payload: dict = {"ok": False, "error": e.message}
        engine = _engine_from_argv(argv)
        if engine is not None:
            payload["engine"] = engine
        print(json.dumps(payload))
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
