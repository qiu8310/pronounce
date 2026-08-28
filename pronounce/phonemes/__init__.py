"""给人看的词典 IPA（不是打分器用的折叠音素表）。"""

from __future__ import annotations

import argparse
import json

from pronounce.phonemes.ipa import ipa_for_text

__all__ = ["add_parser", "ipa_for_text", "run"]


def add_parser(sub: argparse._SubParsersAction) -> None:
    """往 CLI 注册 ``phonemes`` 子命令。"""
    phonemes = sub.add_parser("phonemes", help="dictionary IPA for people to read")
    phonemes.add_argument("--text", required=True)
    phonemes.add_argument("--lang", default="en-us")
    phonemes.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """执行 phonemes：stdout 打一份 JSON（含 ipa 和逐词 rows）。"""
    try:
        payload = ipa_for_text(args.text, lang=args.lang)
        print(
            json.dumps(
                {
                    "ok": True,
                    "command": "phonemes",
                    "text": args.text,
                    "lang": args.lang,
                    # **payload 把 ipa / words 解包进外层字典，避免再套一层。
                    **payload,
                }
            )
        )
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1
