"""从英文文本生成给人看的 IPA（espeak-ng，经 phonemizer）。"""

from __future__ import annotations

from pronounce.common.espeak import ensure_espeak


def ipa_for_text(text: str, lang: str = "en-us") -> dict:
    """把 *text* 转成词典 IPA：一整串，外加按词分行。

    按 ``text.split()`` 每个 token 一行，段落才能和纸面上的词对齐。
    保留重音符号；这是给人读的，不是打分器那套折叠后的音素清单。
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    tokens = text.split()
    ensure_espeak()
    # phonemizer 较重，用到再 import。
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    ipa_list = phonemize(
        tokens,
        language=lang,
        backend="espeak",
        strip=True,
        with_stress=True,           # 保留 ˈ ˌ 等重音标记
        preserve_punctuation=False,
        # Separator：phone/word/syllable 之间插入什么。全空串 = 音素直接连在一起。
        separator=Separator(phone="", word="", syllable=""),
    )
    # 传入单个字符串时 phonemize 可能返回 str 而不是 list，这里统一成列表。
    if isinstance(ipa_list, str):
        ipa_list = [ipa_list]
    words = []
    for tok, ipa in zip(tokens, ipa_list):
        words.append({"word": tok, "ipa": (ipa or "").strip()})
    joined = " ".join(w["ipa"] for w in words if w["ipa"])
    return {"ipa": joined, "words": words}
