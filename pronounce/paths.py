from __future__ import annotations

import os
from pathlib import Path


def models_home() -> Path:
    """Shared models root from ``MODELS_HOME``, or this repo if unset."""
    raw = os.environ.get("MODELS_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    inferred = Path(__file__).resolve().parents[2]
    if (inferred / "llm").is_dir():
        return inferred
    raise RuntimeError(
        "MODELS_HOME is not set; export it to the shared models directory"
    )


def wav2vec2_model(name: str) -> Path:
    return models_home() / "llm" / "wav2vec2" / name
