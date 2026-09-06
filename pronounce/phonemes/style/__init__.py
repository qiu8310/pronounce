"""DJ display-style dispatcher for espeak IPA rewrite."""

from __future__ import annotations

STYLE_LANGS = frozenset({"en-us", "en-gb", "en-gb-x-rp"})


def normalize_style(style: str | None) -> str:
    if style is None or str(style).strip() == "":
        return "none"
    s = str(style).strip()
    if s not in {"none", "dj44", "dj48"}:
        raise ValueError(f"unsupported style: {style!r}")
    return s


def apply_style(ipa: str, *, lang: str, style: str | None) -> str:
    st = normalize_style(style)
    if st == "none":
        return ipa
    if lang not in STYLE_LANGS:
        raise ValueError(
            f"style {st!r} requires lang in {sorted(STYLE_LANGS)}, got {lang!r}"
        )
    if lang == "en-us":
        from pronounce.phonemes.style import en_us

        return en_us.style_word(ipa, style=st)
    if lang == "en-gb":
        from pronounce.phonemes.style import en_gb

        return en_gb.style_word(ipa, style=st)
    from pronounce.phonemes.style import en_gb_x_rp

    return en_gb_x_rp.style_word(ipa, style=st)


def apply_style_phones(
    phones: list[str], *, lang: str, style: str | None
) -> list[str]:
    """Style a phone list per lang (1:1 maps, then dj48 list merge)."""
    st = normalize_style(style)
    if st == "none":
        return list(phones)
    if lang not in STYLE_LANGS:
        raise ValueError(
            f"style {st!r} requires lang in {sorted(STYLE_LANGS)}, got {lang!r}"
        )
    if lang == "en-us":
        from pronounce.phonemes.style import en_us

        return en_us.style_phones(phones, style=st)
    if lang == "en-gb":
        from pronounce.phonemes.style import en_gb

        return en_gb.style_phones(phones, style=st)
    from pronounce.phonemes.style import en_gb_x_rp

    return en_gb_x_rp.style_phones(phones, style=st)
