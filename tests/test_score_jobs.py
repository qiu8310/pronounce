"""score_acoustic / score_phoneme 入参校验（不加载权重）。"""

import tempfile
import unittest
from pathlib import Path

from pronounce.score.jobs import score_acoustic


class TestScoreAcousticValidation(unittest.TestCase):
    def test_empty_text_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as user:
            with tempfile.NamedTemporaryFile(suffix=".wav") as ref:
                with self.assertRaises(ValueError) as ctx:
                    score_acoustic(
                        text="  ",
                        user_wav=user.name,
                        ref_wav=ref.name,
                    )
                self.assertEqual(str(ctx.exception), "text is required")

    def test_missing_ref_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as user:
            with self.assertRaises(FileNotFoundError) as ctx:
                score_acoustic(
                    text="hi",
                    user_wav=user.name,
                    ref_wav="",
                )
            self.assertEqual(str(ctx.exception), "reference audio not found")

    def test_missing_ref_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as user:
            with self.assertRaises(FileNotFoundError) as ctx:
                score_acoustic(
                    text="hi",
                    user_wav=user.name,
                    ref_wav=str(Path("/tmp/no-such-ref.wav")),
                )
            self.assertEqual(str(ctx.exception), "reference audio not found")

    def test_missing_user_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as ref:
            with self.assertRaises(FileNotFoundError) as ctx:
                score_acoustic(
                    text="hi",
                    user_wav=str(Path("/tmp/no-such-user.wav")),
                    ref_wav=ref.name,
                )
            self.assertIn("user audio not found", str(ctx.exception))
