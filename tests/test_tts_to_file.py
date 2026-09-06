import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pronounce.tts import to_file


class TestTtsToFile(unittest.TestCase):
    def test_neither_destination_raises(self):
        with self.assertRaises(ValueError) as ctx:
            to_file(text="hi")
        err = str(ctx.exception).lower()
        self.assertIn("out", err)
        self.assertIn("play", err)

    def test_missing_text_and_ipa_raises(self):
        with self.assertRaises(ValueError) as ctx:
            to_file(out="/tmp/x.wav")
        err = str(ctx.exception).lower()
        self.assertIn("text", err)
        self.assertIn("ipa", err)

    @patch("pronounce.tts.play_audio")
    @patch("pronounce.tts.synthesize", return_value=np.zeros(8, dtype=np.float32))
    def test_play_only(self, synth, play):
        payload = to_file(text="hi", play=True)
        self.assertTrue(payload["ok"])
        self.assertIsNone(payload["out"])
        self.assertTrue(payload["played"])
        synth.assert_called_once()
        play.assert_called_once()
        self.assertEqual(play.call_args.args[1], 24000)

    @patch("pronounce.tts.play_audio")
    @patch("pronounce.tts.synthesize", return_value=np.zeros(8, dtype=np.float32))
    def test_out_only(self, synth, play):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            payload = to_file(text="hi", out=str(path))
            self.assertTrue(path.is_file())
            self.assertEqual(payload["out"], str(path.resolve()))
            self.assertFalse(payload["played"])
            play.assert_not_called()

    @patch("pronounce.tts.play_audio")
    @patch("pronounce.tts.synthesize", return_value=np.zeros(8, dtype=np.float32))
    def test_out_and_play(self, synth, play):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            payload = to_file(text="hi", out=str(path), play=True)
            self.assertTrue(path.is_file())
            self.assertTrue(payload["played"])
            play.assert_called_once()

    @patch("pronounce.tts.play_audio", side_effect=RuntimeError("no device"))
    @patch("pronounce.tts.synthesize", return_value=np.zeros(8, dtype=np.float32))
    def test_play_failure_after_write_keeps_wav(self, synth, play):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            with self.assertRaises(RuntimeError) as ctx:
                to_file(text="hi", out=str(path), play=True)
            self.assertEqual(str(ctx.exception), "no device")
            self.assertTrue(path.is_file())

    @patch("pronounce.tts.play_audio")
    @patch(
        "pronounce.tts.espeak.synthesize_ipa",
        return_value=(np.zeros(8, dtype=np.float32), 22050),
    )
    def test_ipa_play_only(self, ipa_synth, play):
        payload = to_file(ipa="ɪ", play=True)
        self.assertTrue(payload["played"])
        self.assertIsNone(payload["out"])
        self.assertEqual(payload["voice"], "espeak")
        play.assert_called_once()
        self.assertEqual(play.call_args.args[1], 22050)


if __name__ == "__main__":
    unittest.main()
