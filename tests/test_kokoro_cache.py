import unittest
from unittest.mock import MagicMock, patch


class TestKokoroCache(unittest.TestCase):
    def setUp(self):
        from pronounce.tts import kokoro

        kokoro._kmodel = None
        kokoro._pipelines.clear()

    def test_load_tts_constructs_kmodel_once(self):
        from pronounce.tts import kokoro

        fake = MagicMock()
        fake.to.return_value = fake
        fake.eval.return_value = fake
        ctor = MagicMock(return_value=fake)

        with (
            patch.object(kokoro, "_root") as root,
            patch.object(kokoro, "activate_spacy"),
            patch.object(kokoro, "ensure_espeak"),
            patch("kokoro.KModel", ctor),
        ):
            from pathlib import Path

            root.return_value = Path("/tmp/kokoro")
            a = kokoro.load_tts("cpu")
            b = kokoro.load_tts("cpu")
        self.assertIs(a, b)
        self.assertEqual(ctor.call_count, 1)
