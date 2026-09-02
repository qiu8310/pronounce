"""英文 TTS（Kokoro 句子 + espeak 孤立 IPA）。

``to_file`` 是 CLI 和 ``pronounce serve`` 共用的写 wav / JSON 载荷；
``add_parser`` 只负责命令行。
"""

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
    "to_file",
    "voice_path",
]


def to_file(
    *,
    out: str,
    text: str | None = None,
    voice: str = DEFAULT_VOICE,
    lang: str | None = None,
    device: str = "cpu",
    speed: float = 1.0,
    ipa: str | None = None,
) -> dict:
    """合成并写 wav，返回与 CLI / HTTP ``tts`` 相同的成功 JSON。"""
    ipa = (ipa or "").strip() or None
    text = (text or "").strip() or None
    if not out or (not text and not ipa):
        raise ValueError("out and text or ipa are required")
    lang = lang or ("en-gb" if ipa else "en-us")
    text = text or f"/{ipa}/"
    path = Path(out).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if ipa:
        from pronounce.tts.espeak import synthesize_ipa

        audio, native_rate = synthesize_ipa(ipa, lang=lang)
        rate = native_rate
        speed_out = 1.0
        voice_out = "espeak"
    else:
        native_rate = KOKORO_SAMPLE_RATE
        rate = listen_sample_rate(speed, native_rate)
        audio = synthesize(text, voice=voice, lang=lang, device=device)
        speed_out = speed
        voice_out = voice
    import soundfile as sf

    sf.write(str(path), audio, rate)
    payload = {
        "ok": True,
        "command": "tts",
        "text": text,
        "voice": voice_out,
        "lang": lang,
        "out": str(path),
        "speed": speed_out,
        "sample_rate": rate,
        "native_rate": native_rate,
    }
    if ipa:
        payload["ipa"] = ipa
    return payload


def add_parser(sub: argparse._SubParsersAction) -> None:
    """注册 ``tts``：``--out`` 必填，``--text`` 或 ``--ipa`` 二选一。"""
    tts = sub.add_parser("tts", help="synthesize English speech (Kokoro) or one IPA phone (espeak)")
    tts.add_argument("--text", default=None)
    tts.add_argument("--ipa", default=None, help="isolated IPA phone; uses espeak-ng, not Kokoro")
    tts.add_argument("--out", required=True)
    tts.add_argument("--voice", default="af_heart")
    tts.add_argument("--lang", default=None, help="en-us / en-gb; default en-gb with --ipa")
    tts.add_argument("--device", default="cpu")
    tts.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback tempo; 0.8 is slower (lower wav sample rate). Ignored with --ipa",
    )
    tts.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """合成并写 wav，stdout 打 JSON。"""
    try:
        print(
            json.dumps(
                to_file(
                    text=args.text,
                    out=args.out,
                    voice=args.voice,
                    lang=args.lang,
                    device=args.device,
                    speed=args.speed,
                    ipa=args.ipa,
                )
            )
        )
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1
