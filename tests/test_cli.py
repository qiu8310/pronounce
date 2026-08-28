import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pronounce.cli import main
from pronounce.paths import kokoro_model, models_home, spacy_dir, wav2vec2_model


class TestCliGuards(unittest.TestCase):
    def test_acoustic_without_ref_still_needs_user_wav(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["score", "acoustic", "--text", "hi", "--user", "/tmp/no-such-take.wav"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertEqual(data["engine"], "acoustic")
        self.assertIn("user", data["error"].lower())
        self.assertNotIn("requires --ref", data["error"].lower())

    def test_argparse_missing_text_json_exit_1(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["score", "acoustic", "--user", "/tmp/x.wav"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertEqual(data["engine"], "acoustic")
        self.assertIn("error", data)

    def test_schema_prints_fields(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["schema"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("engine", out)
        self.assertIn("ipa", out.lower())

    def test_tts_requires_out(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["tts", "--text", "hi"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertIn("out", data["error"].lower())

    def test_phonemes_requires_text(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["phonemes"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertIn("text", data["error"].lower())

    def test_phonemes_returns_readable_ipa(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["phonemes", "--text", "Hello, how are you?"])
        self.assertEqual(code, 0)
        data = json.loads(buf.getvalue())
        self.assertTrue(data["ok"])
        self.assertIn("ipa", data)
        self.assertIn("words", data)
        self.assertGreater(len(data["words"]), 1)
        self.assertIn("oʊ", data["ipa"] + "".join(w["ipa"] for w in data["words"]))
        self.assertNotIn("O", data["ipa"])

    def test_tts_rejects_invalid_speed(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["tts", "--text", "hi", "--out", "/tmp/x.wav", "--speed", "0"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertNotIn("unrecognized", data["error"].lower())
        self.assertIn("speed", data["error"].lower())

    def test_score_accepts_calibration_flag(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main([
                "score", "phoneme", "--text", "hi",
                "--user", "/tmp/no-such-take.wav",
                "--calibration", "/tmp/cal.json",
            ])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertNotIn("unrecognized", data["error"].lower())
        self.assertIn("user", data["error"].lower())


class TestModelsHome(unittest.TestCase):
    def test_models_home_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"MODELS_HOME": tmp}):
                self.assertEqual(models_home(), Path(tmp).resolve())
                self.assertEqual(
                    wav2vec2_model("wav2vec2-large-960h"),
                    Path(tmp).resolve() / "llm" / "wav2vec2" / "wav2vec2-large-960h",
                )
                self.assertEqual(
                    kokoro_model(),
                    Path(tmp).resolve() / "llm" / "kokoro" / "Kokoro-82M",
                )
                self.assertEqual(
                    spacy_dir(),
                    Path(tmp).resolve() / "llm" / "spacy",
                )


class TestKokoroLang(unittest.TestCase):
    def test_lang_code_aliases(self):
        from pronounce.kokoro import lang_code

        self.assertEqual(lang_code("en-us"), "a")
        self.assertEqual(lang_code("en-gb"), "b")
        with self.assertRaises(ValueError):
            lang_code("es")

    def test_phonemize_rejects_empty(self):
        from pronounce.kokoro import phonemize

        with self.assertRaises(ValueError):
            phonemize("  ")

    def test_listen_sample_rate_slows_playback(self):
        from pronounce.kokoro import KOKORO_SAMPLE_RATE, listen_sample_rate

        self.assertEqual(listen_sample_rate(1.0), KOKORO_SAMPLE_RATE)
        self.assertEqual(listen_sample_rate(0.8), int(round(KOKORO_SAMPLE_RATE * 0.8)))
        with self.assertRaises(ValueError):
            listen_sample_rate(0.0)


if __name__ == "__main__":
    unittest.main()
