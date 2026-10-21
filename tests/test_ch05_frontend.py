from pathlib import Path
import json
import re
import shutil
import subprocess

import pytest


def test_frontend_has_actions_and_ticket_modal():
    html = Path("static/index.html").read_text(encoding="utf-8")
    markers = [
        "case 'actions'",
        "renderActions",
        "transfer_human",
        "openTicketModal",
        "已转接人工客服",
        "工单已创建",
        "fetch('/chat/ticket'",
        "ticketType",
        "ticketDesc",
        "Escape",
    ]
    for marker in markers:
        assert marker in html, marker


def test_tool_badge_keeps_every_tool_name():
    html = Path("static/index.html").read_text(encoding="utf-8")
    assert "function mergeToolBadge" in html, "工具徽章需要累积工具名，而不是覆盖"
    match = re.search(r"function mergeToolBadge[\s\S]*?\n  \}", html)
    assert match, "mergeToolBadge 必须定义在页面脚本中"
    helper = match.group(0)

    node = shutil.which("node")
    if node is None:
        pytest.skip("node 不可用，跳过前端行为测试")

    script = (
        helper
        + "\nconst one = mergeToolBadge(null, 'query_order');"
        + "\nconst two = mergeToolBadge(one.names, 'query_logistics');"
        + "\nconsole.log(JSON.stringify(two));"
    )
    completed = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, encoding="utf-8"
    )
    assert completed.returncode == 0, completed.stderr

    data = json.loads(completed.stdout.strip())
    assert data["names"] == ["query_order", "query_logistics"]
    assert "query_order" in data["text"]
    assert "query_logistics" in data["text"]
