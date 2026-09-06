"""DJ style stress placement (match oral src/lib/ipa.ts)."""

import unittest

from pronounce.phonemes.style.helpers import (
    move_stress,
    relocate_stress,
    strip_monosyllable_stress,
    tokenize_ipa,
)
from pronounce.phonemes.style.en_us import style_word as us_style
from pronounce.phonemes.style.en_gb import style_word as gb_style


class TestTokenizeIpa(unittest.TestCase):
    def test_splits_stress_and_digraphs(self):
        self.assertEqual(tokenize_ipa("trˈiː"), ["tr", "ˈ", "iː"])
        self.assertEqual(tokenize_ipa("əbˈaʊt"), ["ə", "b", "ˈ", "aʊ", "t"])


class TestMoveStress(unittest.TestCase):
    def test_moves_mark_before_onset(self):
        self.assertEqual(
            move_stress(["t", "ˈ", "iː", "tʃ", "ə"]),
            ["ˈ", "t", "iː", "tʃ", "ə"],
        )

    def test_about_medial_onset(self):
        self.assertEqual(
            move_stress(["ə", "b", "ˈ", "aʊ", "t"]),
            ["ə", "ˈ", "b", "aʊ", "t"],
        )


class TestStripMonosyllable(unittest.TestCase):
    def test_strips_single_nucleus(self):
        self.assertEqual(
            strip_monosyllable_stress(["ˈ", "b", "æ", "θ"]),
            ["b", "æ", "θ"],
        )

    def test_keeps_polysyllable(self):
        self.assertEqual(
            strip_monosyllable_stress(["ə", "ˈ", "b", "aʊ", "t"]),
            ["ə", "ˈ", "b", "aʊ", "t"],
        )


class TestRelocateStressString(unittest.TestCase):
    def test_tree_after_merge_loses_mono_stress(self):
        self.assertEqual(relocate_stress("trˈiː"), "triː")

    def test_about(self):
        self.assertEqual(relocate_stress("əbˈaʊt"), "əˈbaʊt")


class TestStyleWordStress(unittest.TestCase):
    def test_us_teacher_stress_before_onset(self):
        self.assertEqual(us_style("tˈiːtʃɚ", style="dj44"), "ˈtiːtʃər")

    def test_us_go_strips_mono_stress(self):
        self.assertEqual(us_style("ɡˈoʊ", style="dj44"), "goʊ")

    def test_gb_bath_strips_mono_stress(self):
        self.assertEqual(gb_style("bˈaθ", style="dj44"), "bæθ")

    def test_us_about_moves_stress(self):
        self.assertEqual(us_style("ɐbˈaʊt", style="dj44"), "əˈbaʊt")


if __name__ == "__main__":
    unittest.main()
