import unittest
from unittest.mock import patch

import numpy as np

from pronounce.tts.espeak import synthesize_ipa


class TestSynthesizeIpa(unittest.TestCase):
    def test_rejects_empty_and_brackets(self):
        with self.assertRaises(ValueError):
            synthesize_ipa("  ")
        with self.assertRaises(ValueError):
            synthesize_ipa("ɪ] extra")

    @patch("soundfile.read", return_value=(np.ones(10, dtype=np.float32), 22050))
    @patch("pronounce.tts.espeak.subprocess.run")
    @patch("pronounce.tts.espeak.espeak_bin", return_value="/usr/bin/espeak-ng")
    def test_writes_padded_mono(self, _bin, run, _read):
        audio, sr = synthesize_ipa("ɪ", lang="en-gb")
        self.assertEqual(sr, 22050)
        self.assertGreater(len(audio), 10)
        args = run.call_args[0][0]
        self.assertIn("[[I]]", args)
        self.assertIn("en-gb", args)


if __name__ == "__main__":
    unittest.main()
