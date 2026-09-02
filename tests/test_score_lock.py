import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class TestScorePhonemeLock(unittest.TestCase):
    def test_score_lock_is_threading_lock(self):
        from pronounce.serve import engines

        self.assertIsInstance(engines._score_lock, type(threading.Lock()))

    def test_concurrent_score_configure_analyze_serialized(self):
        """Two score_phoneme calls must not overlap configure/analyze."""
        from pronounce.serve import engines

        active = 0
        max_active = 0
        gate = threading.Lock()
        fake_result = MagicMock()

        def bump(fn):
            def wrapped(*args, **kwargs):
                nonlocal active, max_active
                with gate:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                try:
                    return fn(*args, **kwargs)
                finally:
                    with gate:
                        active -= 1

            return wrapped

        def configure(_cfg):
            pass

        def analyze(*_a, **_k):
            return fake_result

        wav = np.zeros(8, dtype=np.float32)

        with (
            patch("pronounce.score.jobs.Path") as path_cls,
            patch("soundfile.read", return_value=(wav, 16000)),
            patch(
                "pronounce.common.audio.prepare_waveform",
                side_effect=lambda a, _sr: a,
            ),
            patch(
                "pronounce.score.phoneme.configure",
                side_effect=bump(configure),
            ),
            patch("pronounce.score.phoneme.load_models"),
            patch(
                "pronounce.score.phoneme.analyze",
                side_effect=bump(analyze),
            ),
            patch(
                "pronounce.score.prosody.compute_prosody",
                return_value={"f0": [], "energy": []},
            ),
            patch(
                "pronounce.score.json_out.to_payload",
                return_value={"ok": True, "engine": "phoneme"},
            ),
            patch(
                "pronounce.score.phoneme.AnalyzerConfig",
                side_effect=lambda **kw: MagicMock(**kw),
            ),
            patch(
                "pronounce.paths.wav2vec2_model",
                return_value=Path("/tmp/model"),
            ),
        ):
            path_inst = MagicMock()
            path_inst.is_file.return_value = True
            path_inst.expanduser.return_value = path_inst
            path_inst.resolve.return_value = path_inst
            path_inst.__str__ = lambda self: "/tmp/x.wav"
            path_cls.return_value = path_inst

            errors = []

            def worker(lang):
                try:
                    engines.score_phoneme(
                        text="hi",
                        user_wav="/tmp/u.wav",
                        ref_wav="/tmp/r.wav",
                        lang=lang,
                    )
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            t1 = threading.Thread(target=worker, args=("en-us",))
            t2 = threading.Thread(target=worker, args=("en-gb",))
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(max_active, 1)
