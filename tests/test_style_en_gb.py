import unittest

from pronounce.phonemes.style import apply_style_phones
from pronounce.phonemes.style.en_gb import style_word


class TestEnGbStyle(unittest.TestCase):
    def test_trap_a_to_ae(self):
        self.assertEqual(style_word("bˈaθ", style="dj44"), "bæθ")

    def test_no_us_cloth_rule(self):
        # If short ɔ ever appeared, GB pipeline must not map it; use a synthetic input
        self.assertEqual(style_word("kˈɔf", style="dj44"), "kɔf")

    def test_no_flap_rule(self):
        self.assertEqual(style_word("sˈɪɾi", style="dj44"), "ˈsɪɾi")

    def test_ie_to_ie(self):
        # Monosyllable NEAR: stress stripped after iə→ɪə.
        self.assertEqual(style_word("ˈiə", style="dj44"), "ɪə")

    def test_merge_only_dj48(self):
        self.assertEqual(
            apply_style_phones(["t", "ɹ", "iː"], lang="en-gb", style="dj44"),
            ["t", "r", "iː"],
        )
        self.assertEqual(
            apply_style_phones(["t", "ɹ", "iː"], lang="en-gb", style="dj48"),
            ["tr", "iː"],
        )


if __name__ == "__main__":
    unittest.main()
