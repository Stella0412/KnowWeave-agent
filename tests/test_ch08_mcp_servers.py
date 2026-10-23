"""ch08 · 两个独立 mock MCP Server（logistics / after_sales）行为测试。

两个 Server 是独立进程、纯 mock 数据：不接真实系统、不建业务表、不导入
app.main / app.db / SQLAlchemy。status_code 一律使用服务端内部 ASCII 码，
中文业务标签由可信格式化器（Task 3 formatting.py）在 Agent 侧翻译。

注意：conftest 会导入 app.db.models/base，因此「导入不碰 app.db」这类断言
必须放在干净子进程中验证，不能在当前进程里直接查 sys.modules。
"""

import asyncio
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from mcp_servers.after_sales import make_return_progress, make_warranty
from mcp_servers.logistics import make_tracking

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOGISTICS_CODES = {"pending_pickup", "in_transit", "delivered"}
WARRANTY_CODES = {"in_warranty", "out_of_warranty"}
RETURN_CODES = {"pending", "in_transit", "refunded"}


def _ascii_code(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z_]+", value))


# 1. 三个业务函数：回显对应编号、有效形状、mock=true

def test_make_tracking_echoes_id_and_shape():
    result = make_tracking("SF20261023001", rng=random.Random(1))
    assert result["tracking_no"] == "SF20261023001"
    assert result["mock"] is True
    assert result["carrier"]
    assert result["status_code"] in LOGISTICS_CODES

    events = result["events"]
    assert isinstance(events, list) and events
    for event in events:
        assert event["status_code"] in LOGISTICS_CODES
        datetime.fromisoformat(event["time"])  # ISO-8601 可解析
    # 顶层状态与最新一条轨迹一致
    assert result["status_code"] == events[-1]["status_code"]


def test_make_warranty_echoes_id_and_shape():
    result = make_warranty("智能手环 Pro", rng=random.Random(2))
    assert result["product_name"] == "智能手环 Pro"
    assert result["mock"] is True
    assert isinstance(result["in_warranty"], bool)
    assert result["status_code"] in WARRANTY_CODES
    assert result["status_code"] == ("in_warranty" if result["in_warranty"] else "out_of_warranty")
    datetime.strptime(result["warranty_until"], "%Y-%m-%d")  # 日期字符串


def test_make_return_progress_echoes_id_and_shape():
    result = make_return_progress("RT-20261023-001", rng=random.Random(3))
    assert result["return_no"] == "RT-20261023-001"
    assert result["mock"] is True
    assert result["status_code"] in RETURN_CODES

    progress = result["progress"]
    assert isinstance(progress, list) and progress
    for step in progress:
        assert step["status_code"] in RETURN_CODES
        datetime.fromisoformat(step["time"])
    assert result["status_code"] == progress[-1]["status_code"]


# 2. 固定随机源 / 种子可重复

def test_make_tracking_repeatable_with_seed():
    assert (make_tracking("SF-1", rng=random.Random(42))
            == make_tracking("SF-1", rng=random.Random(42)))


def test_make_warranty_repeatable_with_seed():
    assert (make_warranty("耳机", rng=random.Random(42))
            == make_warranty("耳机", rng=random.Random(42)))


def test_make_return_progress_repeatable_with_seed():
    assert (make_return_progress("R-1", rng=random.Random(42))
            == make_return_progress("R-1", rng=random.Random(42)))


# 3. 物流轨迹时间有序

def test_make_tracking_events_time_ordered():
    for seed in range(20):
        result = make_tracking("T", rng=random.Random(seed))
        times = [datetime.fromisoformat(e["time"]) for e in result["events"]]
        assert times == sorted(times)
        assert all(times[i] < times[i + 1] for i in range(len(times) - 1))


# 4. 导入模块不启动 Server、不 import app.main/app.db/SQLAlchemy、不需要数据库

_IMPORT_GUARD = f"""
import sys, socket
sys.path.insert(0, {str(PROJECT_ROOT)!r})
opened = []
class _RecordingSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        opened.append(args)
        super().__init__(*args, **kwargs)
socket.socket = _RecordingSocket
import mcp_servers.logistics
import mcp_servers.after_sales
assert not opened, opened
assert 'app.main' not in sys.modules, 'imported app.main'
assert 'app.db' not in sys.modules, 'imported app.db'
assert 'sqlalchemy' not in sys.modules, 'imported sqlalchemy'
"""


def test_import_does_not_touch_app_db_or_sockets():
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_GUARD],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr


# 5. status_code 是稳定 ASCII 内部码，不是中文业务标签

def test_status_codes_are_ascii_internal_codes_not_chinese():
    for seed in range(30):
        tracking = make_tracking("T", rng=random.Random(seed))
        assert _ascii_code(tracking["status_code"])
        for event in tracking["events"]:
            assert _ascii_code(event["status_code"])

        warranty = make_warranty("P", rng=random.Random(seed))
        assert _ascii_code(warranty["status_code"])

        ret = make_return_progress("R", rng=random.Random(seed))
        assert _ascii_code(ret["status_code"])
        for step in ret["progress"]:
            assert _ascii_code(step["status_code"])


# 6. MCP 工具委托给业务函数；Server 工厂尊重 host/port 覆盖且不绑定 socket

def test_mcp_tools_delegate_to_business_functions():
    from mcp_servers.after_sales import query_return_progress, query_warranty
    from mcp_servers.logistics import query_logistics

    assert query_logistics("SF-9")["tracking_no"] == "SF-9"
    assert query_warranty("充电器")["product_name"] == "充电器"
    assert query_return_progress("R-9")["return_no"] == "R-9"


def _tool_names(server):
    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_default_servers_register_expected_tools_and_ports():
    from mcp_servers.after_sales import mcp as after_sales_mcp
    from mcp_servers.logistics import mcp as logistics_mcp

    assert logistics_mcp.name == "logistics"
    assert logistics_mcp.settings.host == "127.0.0.1"
    assert logistics_mcp.settings.port == 8101
    assert _tool_names(logistics_mcp) == {"query_logistics"}

    assert after_sales_mcp.name == "after_sales"
    assert after_sales_mcp.settings.host == "127.0.0.1"
    assert after_sales_mcp.settings.port == 8102
    assert _tool_names(after_sales_mcp) == {"query_warranty", "query_return_progress"}


def test_create_server_honors_host_port_overrides_without_binding_socket():
    from mcp_servers.after_sales import create_server as after_sales_create
    from mcp_servers.logistics import create_server as logistics_create

    logistics = logistics_create(host="0.0.0.0", port=9999)
    assert logistics.settings.host == "0.0.0.0"
    assert logistics.settings.port == 9999
    assert logistics.settings.stateless_http is True
    assert logistics.settings.json_response is True

    after_sales = after_sales_create(host="127.0.0.2", port=8888)
    assert after_sales.settings.host == "127.0.0.2"
    assert after_sales.settings.port == 8888
    assert after_sales.settings.stateless_http is True
    assert after_sales.settings.json_response is True
