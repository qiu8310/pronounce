"""中文 TTS（MeloTTS）。

``add_parser`` 注册 ``tts-zh`` 子命令；合成实现在 ``pronounce.tts_zh.melo``。
与英语 ``pronounce.tts`` 并列，不是 ``tts --lang zh``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pronounce.tts.kokoro import listen_sample_rate
from pronounce.tts_zh.melo import DEFAULT_SPEAKER, MELO_SAMPLE_RATE, synthesize

__all__ = ["add_parser", "run"]


def add_parser(sub: argparse._SubParsersAction) -> None:
    """注册 ``tts-zh``：必填 --text / --out，可选设备和播放倍速。"""
    tts = sub.add_parser("tts-zh", help="synthesize Chinese speech with MeloTTS")
    tts.add_argument("--text", required=True)
    tts.add_argument("--out", required=True)
    tts.add_argument("--device", default="cpu")
    tts.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback tempo; 0.8 is slower (lower wav sample rate)",
    )
    tts.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """合成并写 wav，stdout 打 JSON（command 为 tts-zh）。"""
    out = Path(args.out).expanduser().resolve()
    try:
        import contextlib
        import sys

        import soundfile as sf

        rate = listen_sample_rate(args.speed, MELO_SAMPLE_RATE)
        # jieba / transformers 会往 stdout 打日志；合成期间转到 stderr。
        with contextlib.redirect_stdout(sys.stderr):
            audio = synthesize(args.text, device=args.device)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, rate)
        print(
            json.dumps(
                {
                    "ok": True,
                    "command": "tts-zh",
                    "text": args.text,
                    "speaker": DEFAULT_SPEAKER,
                    "lang": "zh",
                    "out": str(out),
                    "speed": args.speed,
                    "sample_rate": rate,
                    "native_rate": MELO_SAMPLE_RATE,
                }
            )
        )
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "command": "tts-zh", "error": str(e)}))
        return 1
