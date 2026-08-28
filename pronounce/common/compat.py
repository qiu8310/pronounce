# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Intel macOS 回退栈的运行时兼容补丁。

Intel Mac（x86_64）上 PyTorch 最高只有 2.2.2 的官方轮子，于是 transformers 也只能
钉在 5 以下。transformers 4.5x 在 torch < 2.6 时拒绝 ``torch.load``（CVE-2025-32434），
除非权重是 safetensors。Mimora 用的几个模型（wav2vec2-large-960h、
wav2vec2-xlsr-53-espeak-cv-ft、nllb-200-distilled-600M）上游只有 ``pytorch_model.bin``，
没有 safetensors，所以 Intel Mac 上会直接加载失败。

这些都是钉死的、可信仓库，不是用户随便给的文件，因此在 Intel Mac 回退栈上
关掉这一道检查。补丁只在 torch < 2.6 时生效；Windows / Linux CUDA / Apple Silicon
（torch >= 2.6）上是空操作。
"""

from __future__ import annotations

import importlib
import logging

# 处理过一次（打过补丁，或判断不需要）后置 True，避免 load_models() 反复跑。
_handled = False


def _torch_below_2_6() -> bool:
    """已装的 torch 是否低于 transformers 要求的 2.6。解析失败当作「够新」。"""
    try:
        import torch

        # 版本串可能是 "2.2.2+cpu"；先去掉 + 后面，再取主.次 两个整数比较。
        major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
        return (major, minor) < (2, 6)
    except Exception:
        return False


def allow_torch_load_for_trusted_models() -> None:
    """关掉 transformers 对 torch.load 的「必须 torch>=2.6」检查。

    仅在 torch < 2.6（Intel Mac 回退）时动手。任何加载 ``.bin`` 的
    ``from_pretrained`` 之前调用一次即可。幂等：重复调用直接返回。
    """
    global _handled  # 要给模块级变量赋值必须声明 global，否则会变成局部变量
    if _handled:
        return
    if not _torch_below_2_6():
        _handled = True  # 新 torch：门禁根本不会触发
        return

    def _noop(*_args, **_kwargs):  # 用来替换 check_torch_load_is_safe
        return None

    patched = False
    # modeling_utils.load_state_dict 调用的是它自己命名空间里绑定的那个名字，
    # 所以要改那个引用；同时也改 import_utils 里的源头，照顾其他调用方。
    for mod_name in ("transformers.modeling_utils", "transformers.utils.import_utils"):
        try:
            # 按字符串动态 import，避免写死 from transformers... 在缺包时直接炸。
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, "check_torch_load_is_safe"):
            # 运行时把函数换成空操作，这就是所谓 monkey-patch（运行时打补丁）。
            mod.check_torch_load_is_safe = _noop
            patched = True

    if patched:
        logging.getLogger(__name__).info(
            "Intel-macOS fallback: allowing torch.load for trusted pinned models "
            "(torch %s < 2.6; models ship only .bin, no safetensors).",
            _installed_torch_version(),
        )
    else:
        # 没找到 check_torch_load_is_safe：可能这版根本没有门禁（加载会成功），
        # 也可能门禁换地方了。打警告，免得 from_pretrained 失败时完全摸不着头脑。
        logging.getLogger(__name__).warning(
            "Intel-macOS fallback: found no check_torch_load_is_safe to disable "
            "(torch %s < 2.6). If loading fails with a torch.load/CVE-2025-32434 "
            "error, the installed transformers version keeps the gate in an "
            "unexpected module.",
            _installed_torch_version(),
        )
    _handled = True


def _installed_torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "unknown"
