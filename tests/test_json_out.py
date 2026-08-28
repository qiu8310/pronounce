import unittest

from pronounce.common import PronunciationResult
from pronounce.score.json_out import to_payload

_ENVELOPE_KEYS = (
    "ok",
    "engine",
    "text",
    "user_wav",
    "ref_wav",
    "score",
    "passed",
    "scored",
    "transcription",
    "feedback",
    "word_errors",
    "words_with_errors",
    "word_diff",
    "reference_words",
    "recognized_units",
    "prosody",
    "phoneme",
    "acoustic",
)


class TestToPayload(unittest.TestCase):
    def test_phoneme_envelope_keys_and_block(self):
        r = PronunciationResult(
            score=80.0, word_errors=[], prosody={}, transcription="hello",
            passed=True, bucket=4, grade="4", ipa_words=[],
        )
        d = to_payload(engine="phoneme", result=r, text="hello", user_wav="/u.wav", ref_wav=None)
        self.assertTrue(d["ok"])
        self.assertEqual(d["engine"], "phoneme")
        self.assertIsNone(d["ref_wav"])
        self.assertEqual(d["prosody"], {})
        self.assertEqual(d["phoneme"]["bucket"], 4)
        self.assertEqual(d["acoustic"], {})
        for key in _ENVELOPE_KEYS:
            self.assertIn(key, d)

    def test_host_prosody_is_passed_through(self):
        r = PronunciationResult(
            score=80.0,
            word_errors=[],
            prosody={},
            transcription="hello",
            passed=True,
            bucket=4,
            grade="4",
            ipa_words=[],
        )
        contours = {"f0": [1.0, 2.0], "energy": [0.5], "ref_f0": [3.0], "ref_energy": [0.2]}
        d = to_payload(
            engine="phoneme", result=r, text="hello", user_wav="/u.wav",
            ref_wav=None, prosody=contours,
        )
        self.assertEqual(d["prosody"], contours)

    def test_acoustic_envelope_swaps_blocks(self):
        r = PronunciationResult(
            score=70.0, word_errors=[], prosody={}, transcription="hello",
            acoustic_distance=1, acoustic_per_step=0.2, acoustic_baseline=0.5,
        )
        d = to_payload(engine="acoustic", result=r, text="hello", user_wav="/u.wav", ref_wav="/r.wav")
        self.assertEqual(d["ref_wav"], "/r.wav")
        self.assertEqual(d["acoustic"]["acoustic_per_step"], 0.2)
        self.assertEqual(d["phoneme"], {})

if __name__ == "__main__":
    unittest.main()
