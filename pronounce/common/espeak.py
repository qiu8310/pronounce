# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""把 phonemizer 指向 wheel 自带的 espeak-ng，两个打分引擎共用。

为什么不需要系统安装
-------------------
``phonemizer`` 通过共享库跟 espeak-ng 说话，不走 ``espeak-ng`` 可执行文件。
``espeakng_loader`` 把库和数据目录打进普通 wheel。注册之后，安装流程里
pip 做不到的那一步就没了。系统安装仍然可用——单独装某个子包、没带
``espeakng_loader`` 时就会退回到系统库——但不是必须的。

为什么引擎要自己注册
-------------------
Kokoro 里的 ``misaki/espeak.py`` 在 import 时也会注册，很长一段时间进程里
只有这一处。不能指望它：Mimora 的 TTS 后端是惰性导入的，一次从不走 Kokoro
的运行（例如用 Supertonic 合成西班牙语）就不会 import misaki，引擎就只能
碰运气用系统里有的东西。本模块是刻意、共享的注册，两个引擎各写一份就没必要了。

分层
----
``pronunciation/*`` 从不 import ``mimora``，这里也只依赖 ``espeakng_loader``
和 ``phonemizer``——而且都在函数内部才 import，这样 ``import`` 本模块不会
拉进第三方代码（和 ``pronounce.common.audio`` 对 librosa 的约束一样）。
缺 ``espeakng_loader`` 是软失败：完整应用会经 kokoro/misaki 间接装上它，
但单独装一个子包时可能没有，那时就靠系统 espeak-ng。

两个值必须一起设
---------------
``set_library()`` 和 ``set_data_path()`` 总是成对设置。只设库看起来能跑，
然后会在 C 库里以访问违例死掉、而不是 Python 异常：``EspeakAPI`` 会把库
拷到临时目录再加载拷贝，数据路径是另传的，所以数据目录不会「就在库旁边」。

诊断
----
注册失败绝不能默默吞掉：否则成功和失败分不清——进程会继续用系统 espeak-ng
（如果有），转写可能和校准用的那套不一样；没有的话更糟。所以打 INFO/WARNING。

也可当脚本跑，``install.py`` 的 ``step_espeak`` 就是这样用的::

    python -m pronounce.common.espeak

必须用 ``-m`` 跑目标环境里的解释器：installer 自己 import 本模块的话，
报告的是 installer 那个解释器，不是用户真正用来打分的那个。
"""

from __future__ import annotations

import logging
import sys

# 单次注册的结果：None 表示还没试过，之后是 True/False。
# 重复调用直接返回上次结果。注册幂等，两线程抢着跑最多赋两次同样的值。
_bundled_registered: bool | None = None


def ensure_espeak() -> bool:
    """向 phonemizer 注册自带的 espeak-ng，只做一次。永不抛异常。

    注册成功返回 True；失败返回 False——phonemizer 会再试
    ``PHONEMIZER_ESPEAK_LIBRARY`` 或系统安装，两者仍可能可用。
    幂等：只有第一次真正干活，之后重复同一结果。
    """
    global _bundled_registered
    if _bundled_registered is not None:
        return _bundled_registered

    log = logging.getLogger(__name__)
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        library = espeakng_loader.get_library_path()
        # get_data_path() 在目录不存在时会抛；必须在改 wrapper 之前先拿到它，
        # 否则会出现「库设了、数据路径没设」的半注册，模块文档里写过那种挂法。
        data_path = espeakng_loader.get_data_path()

        EspeakWrapper.set_library(library)
        EspeakWrapper.set_data_path(data_path)
    except Exception as exc:
        # 不致命，但也不能静默：没有自带库时引擎用系统 espeak-ng，
        # 转写可能和校准那套不同；两边都没有则每个词都会从对比里丢掉。
        log.warning(
            "Bundled espeak-ng could not be registered (%s: %s); falling back "
            "to PHONEMIZER_ESPEAK_LIBRARY or a system espeak-ng. Scores may "
            "differ from the calibrated ones, and phonemization fails "
            "outright if neither is present.",
            type(exc).__name__, exc)
        _bundled_registered = False
        return False

    # set_library() 不检查路径（只是赋值），所以这行日志说的是
    # 「phonemizer 接下来会去加载什么」，不是「已经加载成功」。
    # 真正失败发生在第一次建 backend 的时候。
    log.info("espeak-ng resolves from %s (data %s).", library, data_path)
    _bundled_registered = True
    return True


def resolved_library() -> str | None:
    """phonemizer 实际会用的 espeak 库路径；找不到则 None。

    走 phonemizer 自己的查找，所以能反映三级优先级：上面的注册、
    ``PHONEMIZER_ESPEAK_LIBRARY``、系统搜索——而不只是第一级。
    这才是引擎调用方真正要问的问题；PATH 上有没有 ``espeak-ng`` 可执行文件
    是另一回事，Windows 上两者经常不一致（官方安装器写的是
    ``libespeak-ng.dll``，phonemizer 的系统搜索并不找它）。
    """
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        return str(EspeakWrapper.library())
    except Exception:
        # 什么都找不到是 RuntimeError；没装 phonemizer 是 ImportError。
        # 对调用方来说含义一样。
        return None


def main() -> int:
    """报告当前环境会用哪份 espeak-ng。找到任意一份则退出码 0。

    输出刻意拆开：install.py 会读——**stdout 只有解析到的库路径**
    （没有则为空），说明文字走 stderr。调用方不用解析，人跑命令也能看全貌。
    """
    bundled = ensure_espeak()
    library = resolved_library()

    print(f"Bundled espeak-ng: {'registered' if bundled else 'NOT available'}",
          file=sys.stderr)
    if library is None:
        print("phonemizer finds no espeak-ng library at all. Install "
              "espeakng-loader (pip), or a system espeak-ng and point "
              "PHONEMIZER_ESPEAK_LIBRARY and PHONEMIZER_ESPEAK_DATA_PATH at "
              "it.", file=sys.stderr)
        return 1

    print(f"Resolves to      : {library}", file=sys.stderr)
    print(library)
    return 0


# pragma: no cover 告诉覆盖率工具「别计这一行」——由 install.py 通过 -m 来跑。
if __name__ == "__main__":  # pragma: no cover - exercised via install.py
    raise SystemExit(main())
