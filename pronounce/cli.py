from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_FIELDS_MD = Path(__file__).resolve().parents[1] / "FIELDS.md"

class _ArgError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

class _JsonArgumentParser(argparse.ArgumentParser):
    """Emit stdout JSON failures instead of argparse usage text / exit 2."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise _ArgError(message)

def _cmd_schema(_args: argparse.Namespace) -> int:
    text = _FIELDS_MD.read_text(encoding="utf-8")
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0

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

def main(argv: list[str] | None = None) -> int:
    from pronounce.phonemes import add_parser as add_phonemes
    from pronounce.score import add_parser as add_score
    from pronounce.tts import add_parser as add_tts

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _JsonArgumentParser(prog="pronounce")
    sub = parser.add_subparsers(dest="command", required=True)

    add_score(sub)
    schema = sub.add_parser("schema", help="print FIELDS.md")
    schema.set_defaults(func=_cmd_schema)
    add_tts(sub)
    add_phonemes(sub)

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
