"""CLI 防护测试：缺参、缺文件时应打 JSON 并以码 1 退出，而不是 argparse 的 usage。

不加载打分权重；phonemes 那条会用到本机 espeak。
"""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pronounce.cli import main
from pronounce.paths import (
    kokoro_model,
    melo_bert,
    melo_chinese,
    models_home,
    spacy_dir,
    wav2vec2_model,
)


class TestCliGuards(unittest.TestCase):
    def test_acoustic_without_ref_still_needs_user_wav(self):
        """声学引擎没 --ref 可以自己合成参考，但用户 wav 仍必须存在。"""
        import contextlib
        buf = io.StringIO()
        # redirect_stdout：把 print 接到内存缓冲区，便于断言 JSON，不污染测试输出。
        with contextlib.redirect_stdout(buf):
            code = main(["score", "acoustic", "--text", "hi", "--user", "/tmp/no-such-take.wav"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertEqual(data["engine"], "acoustic")
        self.assertIn("user", data["error"].lower())
        self.assertNotIn("requires --ref", data["error"].lower())

    def test_argparse_missing_text_json_exit_1(self):
        """缺 --text 时走 JSON 失败，退出码 1（不是 argparse 默认的 2）。"""
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
        """schema 子命令打印 FIELDS.md 原文，不是 JSON。"""
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["schema"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("engine", out)
        self.assertIn("ipa", out.lower())
        self.assertIn("tts-zh", out)

    def test_tts_requires_out(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["tts", "--text", "hi"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertIn("out", data["error"].lower())

    def test_tts_zh_requires_out(self):
        """中文 TTS 是独立子命令 tts-zh，缺 --out 时同样 JSON 退出 1。"""
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["tts-zh", "--text", "你好"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertIn("out", data["error"].lower())

    def test_tts_zh_is_not_tts_lang(self):
        """原来的 tts 不接受 zh；中文走 tts-zh，不是 --lang zh。"""
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["tts", "--text", "你好", "--out", "/tmp/x.wav", "--lang", "zh"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertNotEqual(data.get("command"), "tts-zh")

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
        """词典 IPA 应带重音式的 oʊ，而不是大写 O 那种 espeak 内部记号。"""
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
        """speed=0 非法，错误信息应提到 speed，而不是 unrecognized argument。"""
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
        """--calibration 应被 argparse 认下；失败原因应是缺用户 wav，不是未知参数。"""
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
        """MODELS_HOME 指向临时目录时，各权重路径都拼在它下面。"""
        with tempfile.TemporaryDirectory() as tmp:
            # patch.dict：只在 with 块内改 os.environ，退出后恢复，避免污染其他测试。
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
                self.assertEqual(
                    melo_chinese(),
                    Path(tmp).resolve() / "llm" / "melo" / "MeloTTS-Chinese",
                )
                self.assertEqual(
                    melo_bert(),
                    Path(tmp).resolve() / "llm" / "melo" / "chinese-roberta-wwm-ext-large",
                )


class TestKokoroLang(unittest.TestCase):
    def test_lang_code_aliases(self):
        from pronounce.tts import lang_code

        self.assertEqual(lang_code("en-us"), "a")
        self.assertEqual(lang_code("en-gb"), "b")
        with self.assertRaises(ValueError):
            lang_code("es")

    def test_phonemize_rejects_empty(self):
        from pronounce.tts import phonemize

        with self.assertRaises(ValueError):
            phonemize("  ")

    def test_listen_sample_rate_slows_playback(self):
        """speed=0.8 应把写出的采样率降到 0.8 倍（磁带减速），而不是改波形。"""
        from pronounce.tts import KOKORO_SAMPLE_RATE, listen_sample_rate

        self.assertEqual(listen_sample_rate(1.0), KOKORO_SAMPLE_RATE)
        self.assertEqual(listen_sample_rate(0.8), int(round(KOKORO_SAMPLE_RATE * 0.8)))
        with self.assertRaises(ValueError):
            listen_sample_rate(0.0)


class TestMeloZh(unittest.TestCase):
    def test_tts_zh_rejects_invalid_speed(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["tts-zh", "--text", "你好", "--out", "/tmp/x.wav", "--speed", "0"])
        self.assertEqual(code, 1)
        data = json.loads(buf.getvalue())
        self.assertFalse(data["ok"])
        self.assertIn("speed", data["error"].lower())

    def test_synthesize_rejects_empty(self):
        from pronounce.tts_zh.melo import synthesize

        with self.assertRaises(ValueError):
            synthesize("  ")

    def test_missing_checkpoint_mentions_path(self):
        from pronounce.tts_zh.melo import synthesize

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"MODELS_HOME": tmp}):
                with self.assertRaises(FileNotFoundError) as ctx:
                    synthesize("你好")
                self.assertIn("melo", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
