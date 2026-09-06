"""Shared IPA display-style helpers for DJ (dj44/dj48) rewrite."""

from __future__ import annotations

_STRESS = frozenset({"ˈ", "ˌ"})

_RENAME = str.maketrans(
    {
        "ɡ": "g",
        "ɹ": "r",
        "ɛ": "e",
    }
)

_MERGE_FIRST = frozenset({"t", "d"})
_MERGE_SECOND = {
    ("t", "ɹ"): "tr",
    ("t", "r"): "tr",
    ("t", "s"): "ts",
    ("d", "ɹ"): "dr",
    ("d", "r"): "dr",
    ("d", "z"): "dz",
}


def rename_glyphs(s: str) -> str:
    """1:1 glyph renames: ``ɡ→g``, ``ɹ→r``, ``ɛ→e``."""
    return s.translate(_RENAME)


def open_schwa(s: str) -> str:
    """Open near-close central ``ɐ`` to schwa ``ə``."""
    return s.replace("ɐ", "ə")


def split_clusters(s: str) -> str:
    """Split espeak digraphs into textbook phone sequences.

    ``əl`` is treated as ``ə`` + ``l``; ``aɪə`` as ``aɪ`` + ``ə``. Espeak
    already emits these as separate Unicode scalars, so the concatenated string
    is unchanged. Phone-list tokenization in Task 2 relies on these boundaries.
    """
    return s


def map_shared_phone(p: str) -> list[str]:
    """Rename / open-schwa / split one list token. Does not apply lang-specific maps."""
    if p in _STRESS:
        return [p]
    s = rename_glyphs(open_schwa(split_clusters(p)))
    return expand_split_clusters(s)


def expand_split_clusters(s: str) -> list[str]:
    """Turn textbook digraph tokens into separate phones (``əl``, ``aɪə``)."""
    if not s:
        return []
    i = 0
    stress = ""
    while i < len(s) and s[i] in _STRESS:
        stress += s[i]
        i += 1
    rest = s[i:]
    if rest == "əl":
        return [stress + "ə", "l"]
    if rest == "aɪə":
        return [stress + "aɪ", "ə"]
    return [s]


def _phone_core_and_first_stress(p: str) -> tuple[str, str]:
    first = ""
    core: list[str] = []
    for c in p:
        if c in _STRESS:
            if not first:
                first = c
        else:
            core.append(c)
    return "".join(core), first


def merge_dj48_phones(phones: list[str]) -> list[str]:
    """Merge adjacent list phones for dj48 (``t``+``r``/``ɹ`` → ``tr``, etc.).

    Intervening stress-only tokens are skipped for adjacency. Only the first
    stress found (on the first phone, between, or on the second phone) is
    kept, attached to the front of the merged symbol.
    """
    if not phones:
        return []
    out: list[str] = []
    i = 0
    n = len(phones)
    while i < n:
        a = phones[i]
        if a in _STRESS:
            out.append(a)
            i += 1
            continue
        j = i + 1
        between: list[str] = []
        while j < n and phones[j] in _STRESS:
            between.append(phones[j])
            j += 1
        if j < n:
            b = phones[j]
            c1, s1 = _phone_core_and_first_stress(a)
            c2, s2 = _phone_core_and_first_stress(b)
            merged = _MERGE_SECOND.get((c1, c2))
            if merged is not None:
                stress = s1 or (between[0] if between else "") or s2
                out.append(stress + merged)
                i = j + 1
                continue
        out.append(a)
        i += 1
    return out


def merge_dj48(s: str) -> str:
    """Merge adjacent clusters for dj48: ``t+ɹ/r→tr``, ``d+ɹ/r→dr``, ``t+s→ts``, ``d+z→dz``.

    Intervening stress marks ``ˈ`` / ``ˌ`` between the pair are ignored for
    adjacency; the first stress found between the phones attaches to the front
    of the merged symbol (e.g. ``tˈɹiː`` → ``ˈtriː``). Stress before the first
    phone is preserved (``ˈtɹiː`` → ``ˈtriː``).
    """
    if not s:
        return s

    chars = list(s)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c1 = chars[i]
        if c1 in _MERGE_FIRST and i + 1 < len(chars):
            j = i + 1
            stress_between: list[str] = []
            while j < len(chars) and chars[j] in _STRESS:
                stress_between.append(chars[j])
                j += 1
            if j < len(chars):
                merged = _MERGE_SECOND.get((c1, chars[j]))
                if merged is not None:
                    out.append("".join(stress_between) + merged)
                    i = j + 1
                    continue
        out.append(c1)
        i += 1
    return "".join(out)
