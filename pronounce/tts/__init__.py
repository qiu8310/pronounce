"""English TTS (Kokoro)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pronounce.tts.kokoro import (
    DEFAULT_VOICE,
    KOKORO_SAMPLE_RATE,
    lang_code,
    listen_sample_rate,
    phonemize,
    synthesize,
    voice_path,
)

__all__ = [
    "DEFAULT_VOICE",
    "KOKORO_SAMPLE_RATE",
    "add_parser",
    "lang_code",
    "listen_sample_rate",
    "phonemize",
    "run",
    "synthesize",
    "voice_path",
]

def add_parser(sub: argparse._SubParsersAction) -> None:
    tts = sub.add_parser("tts", help="synthesize English speech with Kokoro")
    tts.add_argument("--text", required=True)
    tts.add_argument("--out", required=True)
    tts.add_argument("--voice", default="af_heart")
    tts.add_argument("--lang", default="en-us")
    tts.add_argument("--device", default="cpu")
    tts.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback tempo; 0.8 is slower (lower wav sample rate)",
    )
    tts.set_defaults(func=run)

def run(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    try:
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
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1
