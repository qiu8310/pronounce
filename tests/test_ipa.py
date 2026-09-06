"""词典 IPA：空文本拒绝；段落按空白 token 一行。"""

import unittest

from pronounce.phonemes import ipa_for_text


class TestIpaForText(unittest.TestCase):
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            ipa_for_text("  ")

    def test_paragraph_one_row_per_token(self):
        text = "Hello there. How are you today?"
        out = ipa_for_text(text)
        self.assertEqual(len(out["words"]), len(text.split()))
        self.assertTrue(out["ipa"])
        self.assertNotIn("O", out["ipa"])
        self.assertIn("oʊ", out["ipa"] + "".join(w["ipa"] for w in out["words"]))

    def test_style_none_matches_raw(self):
        raw = ipa_for_text("tree", lang="en-us", style="none")
        omitted = ipa_for_text("tree", lang="en-us")
        self.assertEqual(raw, omitted)

    def test_style_dj48_tree_en_us(self):
        out = ipa_for_text("tree", lang="en-us", style="dj48")
        self.assertIn("tr", out["words"][0]["ipa"])

    def test_style_dj48_rejects_es(self):
        with self.assertRaises(ValueError):
            ipa_for_text("hola", lang="es", style="dj48")

    def test_en_gb_x_rp_phonemizes(self):
        out = ipa_for_text("bath", lang="en-gb-x-rp")
        self.assertTrue(out["ipa"])
        self.assertEqual(out["words"][0]["word"], "bath")


if __name__ == "__main__":
    unittest.main()
