"""把 FIELDS.md（stdout JSON 字段约定）打到 stdout。

``schema`` 不走 JSON 信封：成功时原文输出、退出码 0，方便人读，也方便
宿主把合同当文档打开。FIELDS.md 仍在仓库根，本包只负责打印。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# __file__ 是 pronounce/schema/__init__.py；parents[2] 到仓库根。
_FIELDS_MD = Path(__file__).resolve().parents[2] / "FIELDS.md"

__all__ = ["add_parser", "run"]


def add_parser(sub: argparse._SubParsersAction) -> None:
    """往 CLI 注册 ``schema`` 子命令。"""
    schema = sub.add_parser("schema", help="print FIELDS.md")
    schema.set_defaults(func=run)


def run(_args: argparse.Namespace) -> int:
    """把 FIELDS.md 原样写到 stdout；若文件末尾没有换行则补一个。"""
    text = _FIELDS_MD.read_text(encoding="utf-8")
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0
