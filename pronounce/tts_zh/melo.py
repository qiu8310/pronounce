"""MeloTTS 中文合成，权重来自本机 ``llm/melo/``。运行时不下载。"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

from pronounce.paths import melo_bert, melo_chinese

MELO_SAMPLE_RATE = 44_100
DEFAULT_SPEAKER = "ZH"


def _ckpt_root() -> Path:
    root = melo_chinese()
    if not (root / "config.json").is_file() or not (root / "checkpoint.pth").is_file():
        raise FileNotFoundError(f"MeloTTS-Chinese snapshot missing or incomplete at {root}")
    return root


def _bert_root() -> Path:
    root = melo_bert()
    if not (root / "config.json").is_file() or not (root / "pytorch_model.bin").is_file():
        raise FileNotFoundError(
            f"chinese-roberta-wwm-ext-large snapshot missing or incomplete at {root}"
        )
    return root


def _stub_module(name: str) -> types.ModuleType:
    import importlib.machinery

    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = mod
    return mod


def _stub_optional_imports() -> None:
    """MeloTTS 0.1.1 在 import 时会拉日英依赖；中文合成用不到，缺包就塞空模块。"""
    if "torchaudio" not in sys.modules:
        _stub_module("torchaudio")
    if "cached_path" not in sys.modules:
        cached = _stub_module("cached_path")

        def cached_path(url, *args, **kwargs):
            raise RuntimeError(f"refused download: {url}")

        cached.cached_path = cached_path
    if "MeCab" not in sys.modules:
        mecab = _stub_module("MeCab")
        mecab.Tagger = type("Tagger", (), {"parse": lambda self, text: ""})
    if "g2p_en" not in sys.modules:
        g2p_en = _stub_module("g2p_en")

        class G2p:
            def __call__(self, text):
                return []

        g2p_en.G2p = G2p
    if "pykakasi" not in sys.modules:
        class _Kakasi:
            def setMode(self, *args, **kwargs):
                return None

            def getConverter(self):
                return self

            def do(self, text):
                return text

        pk = _stub_module("pykakasi")
        pk.kakasi = _Kakasi
    if "jamo" not in sys.modules:
        jamo = _stub_module("jamo")
        jamo.hangul_to_jamo = lambda text: list(text)
    if "gruut" not in sys.modules:
        gruut = _stub_module("gruut")
        gruut.sentences = lambda *a, **k: []
        gruut.is_language_supported = lambda language: False
        gruut.get_supported_languages = lambda: []
        gruut.__version__ = "0"
    if "gruut_ipa" not in sys.modules:
        gruut_ipa = _stub_module("gruut_ipa")
        gruut_ipa.IPA = type("IPA", (), {})


def _use_local_weights_and_bert(root: Path) -> None:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    local = str(_bert_root())
    orig_tok = AutoTokenizer.from_pretrained

    def from_pretrained(*args, **kwargs):
        kwargs["local_files_only"] = True
        name = args[0]
        rest = args[1:]
        if not Path(str(name)).is_dir():
            name = local
        return orig_tok(name, *rest, **kwargs)

    AutoTokenizer.from_pretrained = from_pretrained

    from melo import api as melo_api
    from melo import utils as melo_utils
    from melo.text import chinese_bert

    def load_or_download_config(_locale):
        return melo_utils.get_hparams_from_file(str(root / "config.json"))

    def load_or_download_model(_locale, device):
        return torch.load(str(root / "checkpoint.pth"), map_location=device, weights_only=False)

    melo_api.load_or_download_config = load_or_download_config
    melo_api.load_or_download_model = load_or_download_model

    orig = chinese_bert.get_bert_feature

    def get_bert_feature(text, word2ph, device=None, model_id=local):
        # MeloTTS 在 Darwin 上把 device=cpu 改成 mps，但权重仍在 CPU，会炸。
        orig_mps = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False
        try:
            return orig(text, word2ph, device="cpu", model_id=model_id)
        finally:
            torch.backends.mps.is_available = orig_mps

    chinese_bert.get_bert_feature = get_bert_feature


def synthesize(text: str, device: str = "cpu") -> np.ndarray:
    """合成中文 *text*；单声道 float32，44.1 kHz。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    root = _ckpt_root()
    _bert_root()
    _stub_optional_imports()
    _use_local_weights_and_bert(root)
    from melo.api import TTS

    model = TTS(language="ZH", device=device)
    # 0.1.1 会把 ZH 改成 ZH_MIX_EN，从而去拉 bert-base-multilingual；纯中文走 ZH + 本机 RoBERTa。
    model.language = "ZH"
    speaker_id = model.hps.data.spk2id[DEFAULT_SPEAKER]
    audio = model.tts_to_file(text, speaker_id, output_path=None, speed=1.0, quiet=True)
    if audio is None or len(audio) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)
