import unittest

from pronounce.phonemes.style import apply_style_phones
from pronounce.phonemes.style.en_us import style_word


class TestEnUsStyle(unittest.TestCase):
    def test_dj44_city_flap(self):
        self.assertEqual(style_word("sˈɪɾi", style="dj44"), "sˈɪti")

    def test_dj44_car_rhotic(self):
        self.assertEqual(style_word("kˈɑːɹ", style="dj44"), "kˈɑːr")

    def test_dj44_teacher(self):
        self.assertEqual(style_word("tˈiːtʃɚ", style="dj44"), "tˈiːtʃər")

    def test_dj44_go(self):
        self.assertEqual(style_word("ɡˈoʊ", style="dj44"), "gˈəʊ")

    def test_dj44_coffee_cloth(self):
        self.assertEqual(style_word("kˈɔfi", style="dj44"), "kˈɔːfi")

    def test_dj44_tree_no_merge(self):
        # String "trˈiː" is ambiguous (t+r vs atomic tr). Assert merge only
        # via phone lists: dj44 keeps two phones; dj48 merges to tr.
        self.assertEqual(
            apply_style_phones(["t", "ɹ", "iː"], lang="en-us", style="dj44"),
            ["t", "r", "iː"],
        )
        self.assertEqual(
            apply_style_phones(["t", "ɹ", "iː"], lang="en-us", style="dj48"),
            ["tr", "iː"],
        )

    def test_keep_i_and_stress(self):
        self.assertEqual(style_word("hˈæpi", style="dj44"), "hˈæpi")


if __name__ == "__main__":
    unittest.main()
