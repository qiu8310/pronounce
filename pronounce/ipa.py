"""Human-readable English IPA from text (espeak-ng via phonemizer)."""

from __future__ import annotations

from pronounce.common.espeak import ensure_espeak


def ipa_for_text(text: str, lang: str = "en-us") -> dict:
    """Return dictionary IPA for *text*: a joined string plus per-word rows.

    One row per ``text.split()`` token so a paragraph stays aligned with the
    words on the page. Stress marks are kept; this is for people to read, not
    for the scorer's folded inventory.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    tokens = text.split()
    ensure_espeak()
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    ipa_list = phonemize(
        tokens,
        language=lang,
        backend="espeak",
        strip=True,
        with_stress=True,
        preserve_punctuation=False,
        separator=Separator(phone="", word="", syllable=""),
    )
    if isinstance(ipa_list, str):
        ipa_list = [ipa_list]
    words = []
    for tok, ipa in zip(tokens, ipa_list):
        words.append({"word": tok, "ipa": (ipa or "").strip()})
    joined = " ".join(w["ipa"] for w in words if w["ipa"])
    return {"ipa": joined, "words": words}
