"""Dictionary IPA for people to read."""

from __future__ import annotations

import argparse
import json

from pronounce.phonemes.ipa import ipa_for_text

__all__ = ["add_parser", "ipa_for_text", "run"]

def add_parser(sub: argparse._SubParsersAction) -> None:
    phonemes = sub.add_parser("phonemes", help="dictionary IPA for people to read")
    phonemes.add_argument("--text", required=True)
    phonemes.add_argument("--lang", default="en-us")
    phonemes.set_defaults(func=run)

def run(args: argparse.Namespace) -> int:
    try:
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
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1
