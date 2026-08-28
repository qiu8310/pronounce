"""使 ``python -m pronounce`` 能直接跑 CLI。

包目录里的 ``__main__.py`` 是 Python 的约定：对包做 ``-m`` 时会执行这个文件。
"""

from pronounce.cli import main

# SystemExit 把返回码交给操作系统；CLI 约定 0 成功、非 0 失败。
raise SystemExit(main())
