"""Put the local spaCy pipeline on sys.path so misaki's G2P can load it."""

from __future__ import annotations

import importlib
import sys

from pronounce.paths import spacy_dir


def activate() -> None:
    dest = spacy_dir()
    if not dest.is_dir():
        raise FileNotFoundError(
            f"spaCy pipeline missing at {dest}; expected llm/spacy/en_core_web_sm"
        )
    entry = str(dest)
    if entry not in sys.path:
        sys.path.append(entry)
        importlib.invalidate_caches()
