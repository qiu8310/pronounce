"""Score display styling (fold unchanged) — Task 6.

Pure unit tests for the display helpers wired into ``analyze``. These do not
need wav2vec2 / torch: they exercise ``style_ipa_word_entry`` and
``merge_phone_lists_lockstep`` directly, plus assert ``_PHONE_FOLD`` is
untouched. One optional integration test is skipped when the model is absent.
"""

import unittest


class TestFoldTableUntouched(unittest.TestCase):
    def test_fold_entries_preserved(self):
        from pronounce.score.phoneme.speech import _PHONE_FOLD

        self.assertEqual(_PHONE_FOLD.get("æ"), "a")
        self.assertEqual(_PHONE_FOLD.get("ɐ"), "a")
        self.assertEqual(_PHONE_FOLD.get("ɹ"), "r")
        self.assertEqual(_PHONE_FOLD.get("ɾ"), "r")
        self.assertEqual(_PHONE_FOLD.get("ɚ"), "ə")
        self.assertEqual(_PHONE_FOLD.get("oʊ"), "o")


class TestMergePhoneListsLockstep(unittest.TestCase):
    def test_no_merge_keeps_lists(self):
        from pronounce.score.phoneme.speech import merge_phone_lists_lockstep

        exp, heard, ok = merge_phone_lists_lockstep(
            ["k", "æ", "t"], ["k", "a", "t"], [True, False, True]
        )
        self.assertEqual(exp, ["k", "æ", "t"])
        self.assertEqual(heard, ["k", "a", "t"])
        self.assertEqual(ok, [True, False, True])

    def test_merge_tr_and_ok(self):
        from pronounce.score.phoneme.speech import merge_phone_lists_lockstep

        # expected t+r -> tr (merge), heard t+r -> tr (merge), ok ANDs.
        exp, heard, ok = merge_phone_lists_lockstep(
            ["t", "r", "iː"], ["t", "r", "iː"], [True, True, True]
        )
        self.assertEqual(exp, ["tr", "iː"])
        self.assertEqual(heard, ["tr", "iː"])
        self.assertEqual(ok, [True, True])

    def test_merge_ok_ands_false(self):
        from pronounce.score.phoneme.speech import merge_phone_lists_lockstep

        # expected merges t+r, but heard mispronounced s+r (no merge for heard).
        # Lockstep: drive off expected. heard positions concatenate, ok ANDs.
        exp, heard, ok = merge_phone_lists_lockstep(
            ["t", "r", "iː"], ["s", "r", "iː"], [False, True, True]
        )
        self.assertEqual(exp, ["tr", "iː"])
        # heard driven in lockstep: positions 0,1 concatenate -> "sr"
        self.assertEqual(heard, ["sr", "iː"])
        self.assertEqual(ok, [False, True])

    def test_merge_dr(self):
        from pronounce.score.phoneme.speech import merge_phone_lists_lockstep

        exp, heard, ok = merge_phone_lists_lockstep(
            ["d", "r", "iː"], ["d", "r", "iː"], [True, True, True]
        )
        self.assertEqual(exp, ["dr", "iː"])
        self.assertEqual(heard, ["dr", "iː"])
        self.assertEqual(ok, [True, True])

    def test_no_merge_when_not_adjacent_pair(self):
        from pronounce.score.phoneme.speech import merge_phone_lists_lockstep

        # t followed by non-mergeable consonant: no merge.
        exp, heard, ok = merge_phone_lists_lockstep(
            ["t", "k", "æ", "t"], ["t", "k", "a", "t"], [True, True, False, True]
        )
        self.assertEqual(exp, ["t", "k", "æ", "t"])
        self.assertEqual(heard, ["t", "k", "a", "t"])
        self.assertEqual(ok, [True, True, False, True])


class TestStyleIpaWordEntry(unittest.TestCase):
    def test_none_returns_inputs_unchanged(self):
        from pronounce.score.phoneme.speech import style_ipa_word_entry

        exp, heard, ok = style_ipa_word_entry(
            expected_raw=["t", "ɹ", "iː"],
            heard_folded=["t", "r", "iː"],
            ok_folded=[True, True, True],
            lang="en-us",
            style="none",
        )
        self.assertEqual(exp, ["t", "ɹ", "iː"])
        self.assertEqual(heard, ["t", "r", "iː"])
        self.assertEqual(ok, [True, True, True])

    def test_dj44_renames_only(self):
        from pronounce.score.phoneme.speech import style_ipa_word_entry

        exp, heard, ok = style_ipa_word_entry(
            expected_raw=["t", "ɹ", "iː"],
            heard_folded=["t", "r", "iː"],
            ok_folded=[True, True, True],
            lang="en-us",
            style="dj44",
        )
        # ɹ -> r rename on expected; heard already folded r stays r.
        self.assertEqual(exp, ["t", "r", "iː"])
        self.assertEqual(heard, ["t", "r", "iː"])
        self.assertEqual(ok, [True, True, True])

    def test_dj48_merges_tr(self):
        from pronounce.score.phoneme.speech import style_ipa_word_entry

        exp, heard, ok = style_ipa_word_entry(
            expected_raw=["t", "ɹ", "iː"],
            heard_folded=["t", "r", "iː"],
            ok_folded=[True, True, True],
            lang="en-us",
            style="dj48",
        )
        self.assertEqual(exp, ["tr", "iː"])
        self.assertEqual(heard, ["tr", "iː"])
        self.assertEqual(ok, [True, True])

    def test_dj48_merge_ok_ands(self):
        from pronounce.score.phoneme.speech import style_ipa_word_entry

        # Second phone mispronounced: ok[1]=False. After merge, ok=False.
        exp, heard, ok = style_ipa_word_entry(
            expected_raw=["t", "ɹ", "iː"],
            heard_folded=["t", "s", "iː"],
            ok_folded=[True, False, True],
            lang="en-us",
            style="dj48",
        )
        self.assertEqual(exp, ["tr", "iː"])
        # heard per-slot dj44: t, s, iː (no rename for s). Lockstep merge of
        # positions 0,1 -> "ts".
        self.assertEqual(heard, ["ts", "iː"])
        self.assertEqual(ok, [False, True])

    def test_gb_trap_a_to_ae(self):
        from pronounce.score.phoneme.speech import style_ipa_word_entry

        # en-gb: a -> æ (TRAP). Folded heard already has 'a'.
        exp, heard, ok = style_ipa_word_entry(
            expected_raw=["b", "æ", "θ"],  # raw keeps æ
            heard_folded=["b", "a", "θ"],
            ok_folded=[True, True, True],
            lang="en-gb",
            style="dj44",
        )
        # expected raw already has æ; heard folded 'a' -> styled 'æ'.
        self.assertEqual(exp, ["b", "æ", "θ"])
        self.assertEqual(heard, ["b", "æ", "θ"])
        self.assertEqual(ok, [True, True, True])

    def test_us_cloth_short_o(self):
        from pronounce.score.phoneme.speech import style_ipa_word_entry

        # en-us: ɔ (short CLOTH) -> ɔː. Raw expected keeps ɔ.
        exp, heard, ok = style_ipa_word_entry(
            expected_raw=["k", "ɔ", "f", "i"],
            heard_folded=["k", "ɔ", "f", "i"],
            ok_folded=[True, True, True, True],
            lang="en-us",
            style="dj44",
        )
        self.assertEqual(exp, ["k", "ɔː", "f", "i"])
        self.assertEqual(heard, ["k", "ɔː", "f", "i"])

    def test_us_rhotic_split(self):
        from pronounce.score.phoneme.speech import style_ipa_word_entry

        # en-us: ɚ -> ər (split to ə, r in phone list). Folded heard has ə.
        exp, heard, ok = style_ipa_word_entry(
            expected_raw=["t", "iː", "t", "ɚ"],
            heard_folded=["t", "iː", "t", "ə"],
            ok_folded=[True, True, True, True],
            lang="en-us",
            style="dj44",
        )
        # ɚ -> ər -> [ə, r] (split). heard ə -> ə (no split, single phone).
        self.assertEqual(exp, ["t", "iː", "t", "ə", "r"])
        # heard padded with "" to match expected length (split only on expected).
        self.assertEqual(heard, ["t", "iː", "t", "ə", ""])
        # ok replicated: last ok True carries to both ə and r slots on expected,
        # heard padded slot gets the same ok (alignment placeholder).
        self.assertEqual(ok, [True, True, True, True, True])

    def test_rejects_non_en_lang_with_style(self):
        from pronounce.score.phoneme.speech import style_ipa_word_entry

        with self.assertRaises(ValueError):
            style_ipa_word_entry(
                expected_raw=["h", "o", "l", "a"],
                heard_folded=["h", "o", "l", "a"],
                ok_folded=[True, True, True, True],
                lang="es",
                style="dj48",
            )


class TestAnalyzeAcceptsStyle(unittest.TestCase):
    """analyze() must accept style= kwarg with default none (no behavior change)."""

    def test_analyze_has_style_param_default_none(self):
        import inspect

        from pronounce.score.phoneme.speech import analyze

        sig = inspect.signature(analyze)
        self.assertIn("style", sig.parameters)
        self.assertIsNone(sig.parameters["style"].default)


class TestJobsAndServeAcceptStyle(unittest.TestCase):
    def test_jobs_score_phoneme_has_style_param(self):
        import inspect

        from pronounce.score.jobs import score_phoneme

        sig = inspect.signature(score_phoneme)
        self.assertIn("style", sig.parameters)
        self.assertIsNone(sig.parameters["style"].default)

    def test_serve_engines_score_phoneme_has_style_param(self):
        import inspect

        from pronounce.serve.engines import score_phoneme

        sig = inspect.signature(score_phoneme)
        self.assertIn("style", sig.parameters)
        self.assertIsNone(sig.parameters["style"].default)


class TestJobsStyleValidation(unittest.TestCase):
    """jobs.score_phoneme must validate lang+style before model load."""

    def test_rejects_non_en_lang_with_style_before_model_load(self):
        # es + dj48 must raise ValueError without touching wav2vec2.
        from unittest.mock import patch

        from pronounce.score.jobs import score_phoneme

        with patch("pronounce.score.phoneme.load_models") as load_models:
            with self.assertRaises(ValueError) as ctx:
                score_phoneme(
                    text="hola",
                    user_wav="/tmp/u.wav",
                    ref_wav="/tmp/r.wav",
                    lang="es",
                    style="dj48",
                )
            self.assertIn("style", str(ctx.exception).lower())
            # Model load must not have been called — validation is earlier.
            load_models.assert_not_called()

    def test_rejects_invalid_style_value_before_model_load(self):
        from unittest.mock import patch

        from pronounce.score.jobs import score_phoneme

        with patch("pronounce.score.phoneme.load_models") as load_models:
            with self.assertRaises(ValueError):
                score_phoneme(
                    text="hi",
                    user_wav="/tmp/u.wav",
                    ref_wav="/tmp/r.wav",
                    lang="en-us",
                    style="bogus",
                )
            load_models.assert_not_called()

    def test_accepts_explicit_none_with_any_lang(self):
        # explicit "none" on non-en lang must NOT raise from style validation.
        # (It will fail later on missing wav, but not on style.)
        from pronounce.score.jobs import score_phoneme

        try:
            score_phoneme(
                text="hola",
                user_wav="/tmp/no-such.wav",
                ref_wav=None,
                lang="es",
                style="none",
            )
        except ValueError as e:
            self.assertNotIn("style", str(e).lower())
        except FileNotFoundError:
            pass  # expected — wav missing; style validation passed

    def test_omitted_style_omits_key_in_payload(self):
        """jobs path: style=None must NOT echo 'style' in payload (today's envelope)."""
        from unittest.mock import patch

        from pronounce.score.jobs import score_phoneme

        fake_result = type("R", (), {})()
        for attr, val in [
            ("score", 80.0), ("passed", True), ("scored", True),
            ("transcription", "hi"), ("feedback", ""), ("word_errors", []),
            ("words_with_errors", []), ("word_diff", []),
            ("reference_words", []), ("recognized_units", []),
            ("ipa_words", []), ("bucket", -1), ("user_percent", 80.0),
            ("grade", ""), ("grade_value", -1.0), ("per_phone_distance", 0.0),
            ("bad_baseline", 0.4), ("phoneme_score", 80.0), ("recall", 1.0),
            ("good_anchor", 0.0), ("acoustic_distance", 0.0),
            ("acoustic_per_step", 0.0), ("acoustic_baseline", 0.0),
            ("expected_phonemes", []), ("transcribed_phonemes", []),
            ("weak_phonemes", []),
        ]:
            setattr(fake_result, attr, val)

        import numpy as np

        wav = np.zeros(8, dtype=np.float32)
        with (
            patch("pronounce.score.jobs.Path") as path_cls,
            patch("soundfile.read", return_value=(wav, 16000)),
            patch(
                "pronounce.common.audio.prepare_waveform",
                side_effect=lambda a, _sr: a,
            ),
            patch("pronounce.score.phoneme.configure"),
            patch("pronounce.score.phoneme.load_models"),
            patch("pronounce.score.phoneme.analyze", return_value=fake_result),
            patch(
                "pronounce.score.prosody.user_only_prosody",
                return_value={"f0": [], "energy": []},
            ),
        ):
            path_inst = path_cls.return_value
            path_inst.is_file.return_value = True
            path_inst.expanduser.return_value = path_inst
            path_inst.resolve.return_value = path_inst
            path_inst.__str__ = lambda self: "/tmp/u.wav"
            payload = score_phoneme(
                text="hi",
                user_wav="/tmp/u.wav",
                ref_wav=None,
                lang="en-us",
                style=None,
            )
        self.assertNotIn("style", payload)

    def test_explicit_none_echoes_none_in_payload(self):
        """jobs path: explicit style='none' must echo 'style':'none'."""
        from unittest.mock import patch

        from pronounce.score.jobs import score_phoneme

        fake_result = type("R", (), {})()
        for attr, val in [
            ("score", 80.0), ("passed", True), ("scored", True),
            ("transcription", "hi"), ("feedback", ""), ("word_errors", []),
            ("words_with_errors", []), ("word_diff", []),
            ("reference_words", []), ("recognized_units", []),
            ("ipa_words", []), ("bucket", -1), ("user_percent", 80.0),
            ("grade", ""), ("grade_value", -1.0), ("per_phone_distance", 0.0),
            ("bad_baseline", 0.4), ("phoneme_score", 80.0), ("recall", 1.0),
            ("good_anchor", 0.0), ("acoustic_distance", 0.0),
            ("acoustic_per_step", 0.0), ("acoustic_baseline", 0.0),
            ("expected_phonemes", []), ("transcribed_phonemes", []),
            ("weak_phonemes", []),
        ]:
            setattr(fake_result, attr, val)

        import numpy as np

        wav = np.zeros(8, dtype=np.float32)
        with (
            patch("pronounce.score.jobs.Path") as path_cls,
            patch("soundfile.read", return_value=(wav, 16000)),
            patch(
                "pronounce.common.audio.prepare_waveform",
                side_effect=lambda a, _sr: a,
            ),
            patch("pronounce.score.phoneme.configure"),
            patch("pronounce.score.phoneme.load_models"),
            patch("pronounce.score.phoneme.analyze", return_value=fake_result),
            patch(
                "pronounce.score.prosody.user_only_prosody",
                return_value={"f0": [], "energy": []},
            ),
        ):
            path_inst = path_cls.return_value
            path_inst.is_file.return_value = True
            path_inst.expanduser.return_value = path_inst
            path_inst.resolve.return_value = path_inst
            path_inst.__str__ = lambda self: "/tmp/u.wav"
            payload = score_phoneme(
                text="hi",
                user_wav="/tmp/u.wav",
                ref_wav=None,
                lang="en-us",
                style="none",
            )
        self.assertEqual(payload.get("style"), "none")


class TestScorePayloadEchoesStyle(unittest.TestCase):
    """to_payload must echo style at top level (incl. explicit none)."""

    def test_payload_echoes_dj48(self):
        from pronounce.common import PronunciationResult
        from pronounce.score.json_out import to_payload

        r = PronunciationResult(
            score=80.0, word_errors=[], prosody={}, transcription="hi",
            passed=True, bucket=4, grade="4", ipa_words=[],
        )
        d = to_payload(
            engine="phoneme", result=r, text="hi", user_wav="/u.wav",
            ref_wav=None, style="dj48",
        )
        self.assertEqual(d["style"], "dj48")

    def test_payload_echoes_explicit_none(self):
        from pronounce.common import PronunciationResult
        from pronounce.score.json_out import to_payload

        r = PronunciationResult(
            score=80.0, word_errors=[], prosody={}, transcription="hi",
            passed=True, bucket=4, grade="4", ipa_words=[],
        )
        d = to_payload(
            engine="phoneme", result=r, text="hi", user_wav="/u.wav",
            ref_wav=None, style="none",
        )
        self.assertEqual(d["style"], "none")

    def test_payload_omits_style_when_none(self):
        from pronounce.common import PronunciationResult
        from pronounce.score.json_out import to_payload

        r = PronunciationResult(
            score=80.0, word_errors=[], prosody={}, transcription="hi",
            passed=True, bucket=4, grade="4", ipa_words=[],
        )
        d = to_payload(
            engine="phoneme", result=r, text="hi", user_wav="/u.wav", ref_wav=None
        )
        self.assertNotIn("style", d)


class TestAnalyzeStyleNonePreservesBehavior(unittest.TestCase):
    """style=none must not change ipa_words vs today (folded expected/heard/ok)."""

    def test_style_none_keeps_folded_expected(self):
        # Build a minimal ipa_words via build_ipa_words, then style with none.
        from pronounce.score.phoneme.speech import (
            build_ipa_words,
            style_ipa_word_entry,
        )

        tokens = ["tree"]
        groups = [["t", "r", "iː"]]  # folded
        pairs = [("t", "t"), ("r", "r"), ("iː", "iː")]
        spans = [
            {"phone": "t", "t0": 0.0, "t1": 0.1},
            {"phone": "r", "t0": 0.1, "t1": 0.2},
            {"phone": "iː", "t0": 0.2, "t1": 0.4},
        ]
        ipa_words = build_ipa_words(tokens, groups, pairs, spans)
        word = ipa_words[0]
        exp, heard, ok = style_ipa_word_entry(
            expected_raw=word["expected"],
            heard_folded=word["heard"],
            ok_folded=word["ok"],
            lang="en-us",
            style="none",
        )
        self.assertEqual(exp, word["expected"])
        self.assertEqual(heard, word["heard"])
        self.assertEqual(ok, word["ok"])


if __name__ == "__main__":
    unittest.main()
