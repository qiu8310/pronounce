import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pronounce.cli import main
from pronounce.paths import models_home, wav2vec2_model


class TestCliGuards(unittest.TestCase):
    def test_acoustic_requires_ref(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["score", "acoustic", "--text", "hi", "--user", "/tmp/x.wav"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertEqual(data["engine"], "acoustic")
        self.assertIn("ref", data["error"].lower())

    def test_argparse_missing_text_json_exit_1(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["score", "acoustic", "--user", "/tmp/x.wav"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertEqual(data["engine"], "acoustic")
        self.assertIn("error", data)

    def test_schema_prints_fields(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["schema"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("engine", out)


class TestModelsHome(unittest.TestCase):
    def test_models_home_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"MODELS_HOME": tmp}):
                self.assertEqual(models_home(), Path(tmp).resolve())
                self.assertEqual(
                    wav2vec2_model("wav2vec2-large-960h"),
                    Path(tmp).resolve() / "llm" / "wav2vec2" / "wav2vec2-large-960h",
                )


if __name__ == "__main__":
    unittest.main()
