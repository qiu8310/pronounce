# pronounce/tests/test_tts_zh_to_file.py
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pronounce.tts_zh import to_file


class TestTtsZhToFile(unittest.TestCase):
    def test_neither_destination_raises(self):
        with self.assertRaises(ValueError) as ctx:
            to_file(text="你好")
        err = str(ctx.exception).lower()
        self.assertIn("out", err)
        self.assertIn("play", err)

    @patch("pronounce.tts_zh.play_audio")
    @patch("pronounce.tts_zh.synthesize", return_value=np.zeros(8, dtype=np.float32))
    def test_play_only(self, synth, play):
        payload = to_file(text="你好", play=True)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["command"], "tts-zh")
        self.assertIsNone(payload["out"])
        self.assertTrue(payload["played"])
        play.assert_called_once()
        self.assertEqual(play.call_args.args[1], 44100)

    @patch("pronounce.tts_zh.play_audio")
    @patch("pronounce.tts_zh.synthesize", return_value=np.zeros(8, dtype=np.float32))
    def test_out_only(self, synth, play):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zh.wav"
            payload = to_file(text="你好", out=str(path))
            self.assertTrue(path.is_file())
            self.assertFalse(payload["played"])
            play.assert_not_called()

    @patch("pronounce.tts_zh.play_audio")
    @patch("pronounce.tts_zh.synthesize", return_value=np.zeros(8, dtype=np.float32))
    def test_out_and_play(self, synth, play):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zh.wav"
            payload = to_file(text="你好", out=str(path), play=True)
            self.assertTrue(path.is_file())
            self.assertTrue(payload["played"])
            play.assert_called_once()

    @patch("pronounce.tts_zh.play_audio", side_effect=RuntimeError("no device"))
    @patch("pronounce.tts_zh.synthesize", return_value=np.zeros(8, dtype=np.float32))
    def test_play_failure_after_write_keeps_wav(self, synth, play):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zh.wav"
            with self.assertRaises(RuntimeError) as ctx:
                to_file(text="你好", out=str(path), play=True)
            self.assertEqual(str(ctx.exception), "no device")
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
