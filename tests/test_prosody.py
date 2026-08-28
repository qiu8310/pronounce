"""语调曲线：F0 插值、用户+参考四条曲线、无参考时 ref 为空。"""

import unittest

import numpy as np

from pronounce.score import prosody


class TestInterpolateF0(unittest.TestCase):
    def test_fills_gaps(self):
        """无声帧（0）应由两侧有声帧插值填上，整段 > 0。"""
        f0 = np.array([0.0, 100.0, 0.0, 200.0, 0.0])
        out = prosody.interpolate_f0(f0)
        self.assertEqual(len(out), len(f0))
        self.assertTrue((out > 0).all())

    def test_handles_all_silent(self):
        """全静音无法插值，长度不变（值仍是 0）。"""
        f0 = np.zeros(5)
        out = prosody.interpolate_f0(f0)
        self.assertEqual(len(out), 5)


class TestComputeProsody(unittest.TestCase):
    def test_returns_four_contours_as_lists(self):
        """正弦波足够「像语音」让 pyin/RMS 出非空 list，键名固定四条。"""
        sr = prosody.TARGET_SAMPLE_RATE
        t = np.linspace(0, 0.4, int(0.4 * sr), dtype=np.float32)
        user = np.sin(2 * np.pi * 150 * t).astype(np.float32)
        reference = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        out = prosody.compute_prosody(user, sr, reference, sr)
        self.assertEqual(set(out), {"f0", "energy", "ref_f0", "ref_energy"})
        for key, series in out.items():
            self.assertIsInstance(series, list, key)
            self.assertGreater(len(series), 0, key)

    def test_user_only_leaves_ref_empty(self):
        sr = prosody.TARGET_SAMPLE_RATE
        t = np.linspace(0, 0.3, int(0.3 * sr), dtype=np.float32)
        user = np.sin(2 * np.pi * 150 * t).astype(np.float32)
        out = prosody.user_only_prosody(user, sr)
        self.assertEqual(out["ref_f0"], [])
        self.assertEqual(out["ref_energy"], [])
        self.assertGreater(len(out["f0"]), 0)


if __name__ == "__main__":
    unittest.main()
