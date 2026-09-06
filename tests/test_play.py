import unittest
from unittest.mock import patch

import numpy as np

from pronounce.common.play import play_audio, require_destination


class TestRequireDestination(unittest.TestCase):
    def test_out_only_ok(self):
        require_destination("/tmp/a.wav", False)

    def test_play_only_ok(self):
        require_destination(None, True)

    def test_empty_out_without_play_raises(self):
        with self.assertRaises(ValueError) as ctx:
            require_destination("", False)
        err = str(ctx.exception).lower()
        self.assertIn("out", err)
        self.assertIn("play", err)

    def test_none_out_without_play_raises(self):
        with self.assertRaises(ValueError) as ctx:
            require_destination(None, False)
        err = str(ctx.exception).lower()
        self.assertIn("out", err)
        self.assertIn("play", err)


class TestPlayAudio(unittest.TestCase):
    @patch("sounddevice.wait")
    @patch("sounddevice.play")
    def test_plays_array_at_rate_then_waits(self, play, wait):
        audio = np.zeros(4, dtype=np.float32)
        play_audio(audio, 24000)
        play.assert_called_once()
        args, kwargs = play.call_args
        self.assertEqual(kwargs.get("samplerate") or args[1], 24000)
        wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
