import unittest

from pronounce.phonemes.style import apply_style_phones
from pronounce.phonemes.style.en_gb_x_rp import style_word


class TestEnGbXRpStyle(unittest.TestCase):
    def test_bath_keeps_long_a(self):
        self.assertEqual(style_word("bˈɑːθ", style="dj44"), "bˈɑːθ")

    def test_hour_open_schwa(self):
        self.assertEqual(style_word("ˈaʊɐ", style="dj44"), "ˈaʊə")

    def test_no_us_rhotic_split_needed(self):
        # RP non-rhotic teacher-like: no ɚ
        self.assertEqual(style_word("tˈiːtʃə", style="dj44"), "tˈiːtʃə")

    def test_rename_g(self):
        self.assertEqual(style_word("dˈɒɡz", style="dj44"), "dˈɒgz")

    def test_ie_to_ie(self):
        self.assertEqual(style_word("ˈiə", style="dj44"), "ˈɪə")

    def test_trap_short_a_not_converted(self):
        # RP must not reuse GB TRAP a→æ; BATH is already ɑː from G2P.
        self.assertEqual(style_word("bˈaθ", style="dj44"), "bˈaθ")

    def test_no_us_cloth_rule(self):
        self.assertEqual(style_word("kˈɔf", style="dj44"), "kˈɔf")

    def test_no_flap_rule(self):
        self.assertEqual(style_word("sˈɪɾi", style="dj44"), "sˈɪɾi")

    def test_merge_only_dj48(self):
        self.assertEqual(
            apply_style_phones(["t", "ɹ", "iː"], lang="en-gb-x-rp", style="dj44"),
            ["t", "r", "iː"],
        )
        self.assertEqual(
            apply_style_phones(["t", "ɹ", "iː"], lang="en-gb-x-rp", style="dj48"),
            ["tr", "iː"],
        )


if __name__ == "__main__":
    unittest.main()
