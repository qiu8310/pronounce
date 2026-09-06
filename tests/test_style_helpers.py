import unittest

from pronounce.phonemes.style import apply_style, normalize_style
from pronounce.phonemes.style.helpers import (
    merge_dj48,
    open_schwa,
    rename_glyphs,
    split_clusters,
)


class TestNormalizeStyle(unittest.TestCase):
    def test_omit_and_none(self):
        self.assertEqual(normalize_style(None), "none")
        self.assertEqual(normalize_style("none"), "none")
        self.assertEqual(normalize_style(""), "none")
        self.assertEqual(normalize_style("  "), "none")

    def test_valid(self):
        self.assertEqual(normalize_style("dj44"), "dj44")
        self.assertEqual(normalize_style("dj48"), "dj48")

    def test_invalid(self):
        with self.assertRaises(ValueError):
            normalize_style("dj99")


class TestHelpers(unittest.TestCase):
    def test_rename(self):
        self.assertEqual(rename_glyphs("ɡɹɛɪ"), "greɪ")

    def test_open_schwa(self):
        self.assertEqual(open_schwa("ɐbˈaʊt"), "əbˈaʊt")

    def test_split_clusters_preserves_string(self):
        # əl and aɪə are already ə+l and aɪ+ə in espeak output; the string
        # form is unchanged. Task 2 phone-list APIs use these boundaries.
        cases = ("əl", "aɪə", "həlˈoʊ", "aɪəˈdiə", "hˈæpi")
        for inp in cases:
            with self.subTest(inp=inp):
                self.assertEqual(split_clusters(inp), inp)

    def test_merge_dj48(self):
        self.assertEqual(merge_dj48("tɹˈiː"), "trˈiː")
        self.assertEqual(merge_dj48("tˈɹiː"), "ˈtriː")
        self.assertEqual(merge_dj48("ˈtɹiː"), "ˈtriː")
        self.assertEqual(merge_dj48("kæts"), "kæts")
        self.assertEqual(merge_dj48("dˈzɑː"), "ˈdzɑː")


class TestApplyStyleGuard(unittest.TestCase):
    def test_none_passthrough_any_lang(self):
        self.assertEqual(apply_style("xˈa", lang="es", style="none"), "xˈa")
        self.assertEqual(apply_style("xˈa", lang="es", style=None), "xˈa")

    def test_dj48_rejects_non_en(self):
        with self.assertRaises(ValueError) as ctx:
            apply_style("xˈa", lang="es", style="dj48")
        self.assertIn("style", str(ctx.exception))

    def test_dj44_rejects_non_en(self):
        with self.assertRaises(ValueError) as ctx:
            apply_style("xˈa", lang="es", style="dj44")
        self.assertIn("style", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
