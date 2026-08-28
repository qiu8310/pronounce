"""把本机 spaCy 管道目录挂到 ``sys.path``，让 misaki 的 G2P 能 load 到它。

Kokoro / misaki 用 ``import en_core_web_sm`` 这种包名加载 spaCy；
权重不在 site-packages 里，而在 ``$MODELS_HOME/llm/spacy``，所以要把该目录
插进模块搜索路径。
"""

from __future__ import annotations

import importlib
import sys

from pronounce.paths import spacy_dir


def activate() -> None:
    """确保 ``llm/spacy`` 在 ``sys.path`` 上。目录不存在则立刻报错。"""
    dest = spacy_dir()
    if not dest.is_dir():
        raise FileNotFoundError(
            f"spaCy pipeline missing at {dest}; expected llm/spacy/en_core_web_sm"
        )
    entry = str(dest)
    if entry not in sys.path:
        sys.path.append(entry)
        # 改过 sys.path 后清掉 import 缓存，避免之前「找不到包」的失败结果还留着。
        importlib.invalidate_caches()
