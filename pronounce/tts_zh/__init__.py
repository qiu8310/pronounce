"""中文 TTS（MeloTTS）。

``add_parser`` 注册 ``tts-zh`` 子命令；合成实现在 ``pronounce.tts_zh.melo``。
与英语 ``pronounce.tts`` 并列，不是 ``tts --lang zh``。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from pronounce.common.play import play_audio, require_destination
from pronounce.tts.kokoro import listen_sample_rate
from pronounce.tts_zh.melo import DEFAULT_SPEAKER, MELO_SAMPLE_RATE, synthesize

__all__ = ["add_parser", "run", "to_file"]


def to_file(
    *,
    text: str,
    out: str | None = None,
    play: bool = False,
    device: str = "cpu",
    speed: float = 1.0,
) -> dict:
    """合成并可选写 wav / 播放，返回与 CLI ``tts-zh`` 相同的成功 JSON。"""
    require_destination(out, play)
    rate = listen_sample_rate(speed, MELO_SAMPLE_RATE)
    # jieba / transformers 会往 stdout 打日志；合成期间转到 stderr。
    with contextlib.redirect_stdout(sys.stderr):
        audio = synthesize(text, device=device)
    out_path = (out or "").strip() or None
    written = None
    if out_path:
        path = Path(out_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        sf.write(str(path), audio, rate)
        written = str(path)
    if play:
        play_audio(audio, rate)
    return {
        "ok": True,
        "command": "tts-zh",
        "text": text,
        "speaker": DEFAULT_SPEAKER,
        "lang": "zh",
        "out": written,
        "played": play,
        "speed": speed,
        "sample_rate": rate,
        "native_rate": MELO_SAMPLE_RATE,
    }


def add_parser(sub: argparse._SubParsersAction) -> None:
    """注册 ``tts-zh``：必填 --text，--out 或 --play，可选设备和播放倍速。"""
    tts = sub.add_parser("tts-zh", help="synthesize Chinese speech with MeloTTS")
    tts.add_argument("--text", required=True)
    tts.add_argument("--out", default=None)
    tts.add_argument("--play", action="store_true")
    tts.add_argument("--device", default="cpu")
    tts.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback tempo; 0.8 is slower (lower wav sample rate)",
    )
    tts.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """合成并可选写 wav / 播放，stdout 打 JSON（command 为 tts-zh）。"""
    try:
        print(
            json.dumps(
                to_file(
                    text=args.text,
                    out=args.out,
                    play=args.play,
                    device=args.device,
                    speed=args.speed,
                )
            )
        )
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "command": "tts-zh", "error": str(e)}))
        return 1
