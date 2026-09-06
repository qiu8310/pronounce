"""US English DJ display-style pipeline."""

from __future__ import annotations

import re

from pronounce.phonemes.style import helpers as h

_STRESS = frozenset({"ˈ", "ˌ"})
# CLOTH short ɔ → ɔː; do not lengthen CHOICE ɔɪ or already-long ɔː.
_SHORT_CLOTH = re.compile(r"ɔ(?![ːɪ])")


def _us_pre(s: str) -> str:
    s = s.replace("ɚ", "ər")
    s = s.replace("ɝ", "ɜːr")
    s = s.replace("ɑːɹ", "ɑːr")
    s = s.replace("ɔːɹ", "ɔːr")
    s = s.replace("ɪɹ", "ɪr")
    s = s.replace("ɛɹ", "er")
    s = s.replace("ʊɹ", "ʊr")
    s = s.replace("ɾ", "t")
    # Keep GOAT oʊ (do not Anglicize to əʊ).
    s = _SHORT_CLOTH.sub("ɔː", s)
    return s


def _shared(s: str) -> str:
    return h.rename_glyphs(h.open_schwa(h.split_clusters(s)))


def style_word(ipa: str, *, style: str) -> str:
    s = _shared(_us_pre(ipa))
    if style == "dj48":
        s = h.merge_dj48(s)
    return h.relocate_stress(s)


def _core(p: str) -> str:
    return "".join(c for c in p if c not in _STRESS)


def _map_us_phone(p: str) -> list[str]:
    if p in _STRESS:
        return [p]
    orig_core = _core(p)
    s = _shared(_us_pre(p))
    if orig_core in {"ɚ", "ɝ"} and s.endswith("r") and len(s) > 1:
        return [s[:-1], "r"]
    if orig_core.endswith("ɹ") and len(orig_core) > 1 and s.endswith("r"):
        return [s[:-1], "r"]
    return h.expand_split_clusters(s)


def style_phones(phones: list[str], *, style: str) -> list[str]:
    out: list[str] = []
    for p in phones:
        out.extend(_map_us_phone(p))
    if style == "dj48":
        out = h.merge_dj48_phones(out)
    return h.relocate_stress_phones(out)
