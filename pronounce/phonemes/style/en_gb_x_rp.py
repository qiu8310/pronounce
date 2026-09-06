"""RP English DJ display-style pipeline."""

from __future__ import annotations

from pronounce.phonemes.style import helpers as h


def _rp_pre(s: str) -> str:
    # GB-family glyph fix for NEAR-like iə; do not TRAP-map a→æ (BATH is ɑː).
    return s.replace("iə", "ɪə")


def style_word(ipa: str, *, style: str) -> str:
    s = h.rename_glyphs(h.open_schwa(h.split_clusters(_rp_pre(ipa))))
    if style == "dj48":
        s = h.merge_dj48(s)
    return s


def style_phones(phones: list[str], *, style: str) -> list[str]:
    out: list[str] = []
    for p in phones:
        out.extend(h.map_shared_phone(_rp_pre(p)))
    if style == "dj48":
        out = h.merge_dj48_phones(out)
    return out
