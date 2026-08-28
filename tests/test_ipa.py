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


if __name__ == "__main__":
    unittest.main()
