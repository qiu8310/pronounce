"""pronounce 命令行入口：解析子命令并分发给 score / schema / tts / phonemes。

成功时子命令自己往 stdout 打一份 JSON；失败也打 JSON（``ok: false``），
退出码 1。不用 argparse 默认的 usage 文本 + exit 2，方便被 mimora 等宿主解析。
``schema`` 例外：打印 FIELDS.md 原文，不走 JSON 信封。
"""

from __future__ import annotations  # 推迟求值类型注解，才能写 list[str] | None 这种写法

import argparse
import json
import sys


class _ArgError(Exception):
    """把 argparse 的错误从「直接退出」改成「可捕获的异常」。"""

    def __init__(self, message: str) -> None:
        self.message = message


class _JsonArgumentParser(argparse.ArgumentParser):
    """参数解析失败时抛 _ArgError，而不是打印 usage 并以码 2 退出。"""

    def error(self, message: str) -> None:  # type: ignore[override]
        # 父类 error() 标注为 NoReturn（永不返回）；我们改成抛异常，所以用
        # type: ignore 告诉类型检查器「这里故意不符合父类签名」。
        raise _ArgError(message)


def _engine_from_argv(argv: list[str]) -> str | None:
    """从原始 argv 里抠出 score 后面的引擎名（phoneme / acoustic）。

    解析失败时还拿不到 args.engine，只能自己扫一遍命令行，
    好在错误 JSON 里带上 ``engine`` 字段。
    """
    try:
        score_idx = argv.index("score")
    except ValueError:
        return None
    if score_idx + 1 >= len(argv):
        return None
    engine = argv[score_idx + 1]
    if engine in ("phoneme", "acoustic"):
        return engine
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI 主函数。

    argv 为 None 时用 sys.argv[1:]（真正从命令行调用）；测试里会传入假参数列表。
    返回进程退出码。
    """
    # 延迟导入：避免一 import cli 就把 score/tts 的重依赖拉进来。
    from pronounce.phonemes import add_parser as add_phonemes
    from pronounce.schema import add_parser as add_schema
    from pronounce.score import add_parser as add_score
    from pronounce.tts import add_parser as add_tts

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _JsonArgumentParser(prog="pronounce")
    # dest="command" 把选中的子命令名写到 args.command；required=True 表示必须选一个。
    sub = parser.add_subparsers(dest="command", required=True)

    add_score(sub)
    add_schema(sub)
    add_tts(sub)
    add_phonemes(sub)

    try:
        args = parser.parse_args(argv)
    except _ArgError as e:
        payload: dict = {"ok": False, "error": e.message}
        engine = _engine_from_argv(argv)
        if engine is not None:
            payload["engine"] = engine
        print(json.dumps(payload))
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
