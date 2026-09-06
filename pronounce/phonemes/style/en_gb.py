"""GB English DJ display-style pipeline."""

from __future__ import annotations

import re

from pronounce.phonemes.style import helpers as h

# TRAP a → æ; do not change PRICE aɪ, MOUTH aʊ, or PALM/START ɑ/ɑː.
_TRAP_A = re.compile(r"a(?![ɪʊ])")


def _gb_pre(s: str) -> str:
    s = _TRAP_A.sub("æ", s)
    s = s.replace("iə", "ɪə")
    return s


def style_word(ipa: str, *, style: str) -> str:
    s = h.rename_glyphs(h.open_schwa(h.split_clusters(_gb_pre(ipa))))
    if style == "dj48":
        s = h.merge_dj48(s)
    return s


def style_phones(phones: list[str], *, style: str) -> list[str]:
    out: list[str] = []
    for p in phones:
        out.extend(h.map_shared_phone(_gb_pre(p)))
    if style == "dj48":
        out = h.merge_dj48_phones(out)
    return out
