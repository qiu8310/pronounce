from __future__ import annotations

import argparse
import json
import sys
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
        from pronounce.kokoro import KOKORO_SAMPLE_RATE, synthesize

        audio = synthesize(
            args.text, voice=args.voice, lang=args.lang, device=args.device
        )
        import soundfile as sf

        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, KOKORO_SAMPLE_RATE)
        print(
            json.dumps(
                {
                    "ok": True,
                    "command": "tts",
                    "text": args.text,
                    "voice": args.voice,
                    "lang": args.lang,
                    "out": str(out),
                    "sample_rate": KOKORO_SAMPLE_RATE,
                }
            )
        )
        return 0
    except Exception as e:
        return _fail_plain(str(e))


def _cmd_phonemes(args: argparse.Namespace) -> int:
    try:
        from pronounce.kokoro import phonemize

        phonemes = phonemize(args.text, lang=args.lang)
        print(
            json.dumps(
                {
                    "ok": True,
                    "command": "phonemes",
                    "text": args.text,
                    "lang": args.lang,
                    "phonemes": phonemes,
                }
            )
        )
        return 0
    except Exception as e:
        return _fail_plain(str(e))


def _cmd_score(args: argparse.Namespace) -> int:
    engine = args.engine
    if engine == "acoustic" and not args.ref:
        return _fail(engine, "the acoustic engine requires --ref")

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

    if not user_path.is_file():
        return _fail(engine, f"user audio not found: {user_path}")
    if ref_path is not None and not ref_path.is_file():
        return _fail(engine, f"reference audio not found: {ref_path}")

    try:
        from pronounce.common.audio import TARGET_SAMPLE_RATE, prepare_waveform
        from pronounce.json_out import to_payload

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

        configure(
            AnalyzerConfig(
                model_name=model_name,
                device=args.device,
                espeak_language=args.lang,
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
        payload = to_payload(
            engine=engine,
            result=result,
            text=args.text,
            user_wav=str(user_path),
            ref_wav=str(ref_path) if ref_path is not None else None,
        )
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
    score.set_defaults(func=_cmd_score)

    schema = sub.add_parser("schema", help="print FIELDS.md")
    schema.set_defaults(func=_cmd_schema)

    tts = sub.add_parser("tts", help="synthesize English speech with Kokoro")
    tts.add_argument("--text", required=True)
    tts.add_argument("--out", required=True)
    tts.add_argument("--voice", default="af_heart")
    tts.add_argument("--lang", default="en-us")
    tts.add_argument("--device", default="cpu")
    tts.set_defaults(func=_cmd_tts)

    phonemes = sub.add_parser("phonemes", help="grapheme-to-phoneme via Kokoro/misaki")
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
