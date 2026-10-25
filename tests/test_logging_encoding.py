"""日志编码回归：stderr 重定向到文件时，中文 trace 必须是 UTF-8。"""

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_turn_log_is_utf8_when_stderr_redirected(tmp_path):
    """Windows 下 stderr 重定向到文件默认用 locale(cp936)，中文 trace 会写坏。"""
    target = tmp_path / "stderr.log"
    code = (
        "import logging\n"
        "import app.main  # noqa: F401  导入即完成日志配置\n"
        "logging.getLogger('knowweave.turn').info('中文追踪 订单 1001')\n"
    )
    # 清掉可能掩盖 locale 编码的开关，复现真实的「重定向到文件」场景。
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONIOENCODING", "PYTHONUTF8", "PYTHONLEGACYWINDOWSSTDIO"}
    }
    with target.open("wb") as stderr:
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            check=True,
        )
    text = target.read_bytes().decode("utf-8")
    assert "中文追踪 订单 1001" in text
