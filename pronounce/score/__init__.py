"""给口语跟读打分：音素引擎或声学引擎。

``add_parser`` 把 ``score`` 子命令挂到 CLI；``run`` 负责读 wav、可选地用
Kokoro 合成参考音、调对应引擎的 ``analyze``，再把结果打成 stdout JSON。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from pronounce.paths import wav2vec2_model


def add_parser(sub: argparse._SubParsersAction) -> None:
    """注册 ``score {phoneme|acoustic}`` 及其参数。"""
    score = sub.add_parser("score", help="score a spoken take")
    score.add_argument("engine", choices=("phoneme", "acoustic"))
    score.add_argument("--text", default=None)
    score.add_argument("--ipa", default=None, help="isolated IPA phone (phoneme engine; skips G2P)")
    score.add_argument("--user", required=True)
    score.add_argument("--ref", default=None)
    score.add_argument("--lang", default="en-us")
    score.add_argument("--device", default="cpu")
    score.add_argument(
        "--voice",
        default="af_heart",
        help="Kokoro voice when --ref is omitted (acoustic)",
    )
    score.add_argument("--calibration", default=None, help="per-user calibration.json")
    # dest="user_name"：命令行是 --user-name，Python 里属性名不能带连字符。
    score.add_argument("--user-name", default="", dest="user_name")
    score.set_defaults(func=run)


def _fail(engine: str, error: str) -> int:
    """失败时打 JSON 并返回退出码 1。"""
    print(json.dumps({"ok": False, "engine": engine, "error": error}))
    return 1


def _check_model_dir(engine: str, model_dir: str) -> str | None:
    """检查是否为可用的 Hugging Face ``from_pretrained`` 根目录；OK 则返回 None。"""
    root = Path(model_dir)
    if not root.is_dir() or not (root / "config.json").is_file():
        return (
            f"model directory missing or incomplete at {model_dir}; "
            f"expected a from_pretrained root under llm/wav2vec2/"
        )
    return None


def _load_wav(path: str):
    """读 wav，返回 (ndarray, 采样率)。always_2d=False 让单声道保持一维。"""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    return data, int(sr)


def _synthesize_ref(args: argparse.Namespace) -> Path:
    """声学引擎没给 --ref 时，用 Kokoro 合成一段参考音，写到临时 wav。"""
    import soundfile as sf

    from pronounce.tts import KOKORO_SAMPLE_RATE, synthesize

    audio = synthesize(
        args.text, voice=args.voice, lang=args.lang, device=args.device
    )
    # mkstemp 返回 (fd, path)；立刻 close(fd)，后面用 soundfile 按路径写，
    # 避免文件描述符一直占着。
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="pronounce-ref-")
    os.close(fd)
    path = Path(tmp)
    sf.write(str(path), audio, KOKORO_SAMPLE_RATE)
    return path


def run(args: argparse.Namespace) -> int:
    """跑一次打分。成功打印 payload JSON，返回 0。"""
    engine = args.engine
    if args.ipa and engine != "phoneme":
        return _fail(engine, "--ipa is only valid with the phoneme engine")
    if not args.text and not args.ipa:
        return _fail(engine, "text or ipa is required")

    model_name = str(
        wav2vec2_model(
            "wav2vec2-xlsr-53-espeak-cv-ft"
            if engine == "phoneme"
            else "wav2vec2-large-960h"
        )
    )
    model_err = _check_model_dir(engine, model_name)
    if model_err is not None:
        return _fail(engine, model_err)

    user_path = Path(args.user).resolve()
    ref_path = Path(args.ref).resolve() if args.ref else None
    ref_generated = False

    if not user_path.is_file():
        return _fail(engine, f"user audio not found: {user_path}")
    if ref_path is not None and not ref_path.is_file():
        return _fail(engine, f"reference audio not found: {ref_path}")

    try:
        if engine == "phoneme":
            from pronounce.score.jobs import score_phoneme

            payload = score_phoneme(
                text=args.text,
                user_wav=str(user_path),
                ref_wav=str(ref_path) if ref_path is not None else None,
                lang=args.lang,
                device=args.device,
                ipa=args.ipa,
                calibration=args.calibration,
                user_name=args.user_name or "",
            )
            print(json.dumps(payload))
            return 0

        from pronounce.common.audio import TARGET_SAMPLE_RATE, prepare_waveform
        from pronounce.score.acoustic import AnalyzerConfig, analyze, configure, load_models
        from pronounce.score.json_out import to_payload
        from pronounce.score.prosody import compute_prosody

        if ref_path is None:
            ref_path = _synthesize_ref(args)
            ref_generated = True

        user_audio, user_sr = _load_wav(str(user_path))
        user_audio = prepare_waveform(user_audio, user_sr)
        reference_audio, reference_sr = _load_wav(str(ref_path))
        reference_audio = prepare_waveform(reference_audio, reference_sr)

        cal = Path(args.calibration).expanduser().resolve() if args.calibration else None
        configure(
            AnalyzerConfig(
                model_name=model_name,
                device=args.device,
                espeak_language=args.lang,
                user_name=args.user_name or "",
                calibration_file=cal,
            )
        )
        load_models()
        result = analyze(
            user_audio,
            args.text,
            reference_audio=reference_audio,
            user_sr=TARGET_SAMPLE_RATE,
            reference_sr=TARGET_SAMPLE_RATE,
        )
        contours = compute_prosody(
            user_audio, TARGET_SAMPLE_RATE, reference_audio, TARGET_SAMPLE_RATE
        )
        payload = to_payload(
            engine=engine,
            result=result,
            text=args.text,
            user_wav=str(user_path),
            ref_wav=str(ref_path),
            prosody=contours,
        )
        if ref_generated:
            payload["ref_generated"] = True
        print(json.dumps(payload))
        return 0
    except Exception as e:
        return _fail(engine, str(e))
