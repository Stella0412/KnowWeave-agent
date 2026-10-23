"""ch08 · MCP 动态发现与授权适配（MCPToolProvider / create_mcp_client）测试。

fake-client 测试用真实 PolicyStore + tmp_path 权限文件；真实 SDK Server 集成测试
单独启动两个独立进程，只清理本测试启动的进程，不依赖 fake client。
"""

import asyncio
import json
import socket
import sys

import httpx
import pytest
from langchain_core.tools import ToolException

from app.tools.contracts import ExecutionContext, TransientToolError

LOGISTICS_CODES = {"pending_pickup", "in_transit", "delivered"}
WARRANTY_CODES = {"in_warranty", "out_of_warranty"}
RETURN_CODES = {"pending", "in_transit", "refunded"}

LOGISTICS_TEXT = {"pending_pickup": "待揽收", "in_transit": "运输中", "delivered": "已签收"}
WARRANTY_TEXT = {"in_warranty": "在保", "out_of_warranty": "过保"}
RETURN_TEXT = {"pending": "待处理", "in_transit": "退回中", "refunded": "已退款"}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_policy(path, mcp):
    path.write_text(json.dumps({"mcp": mcp}, ensure_ascii=False), encoding="utf-8")
    return path


class FakeTool:
    """StructuredTool-like：暴露 name/description/args_schema 与异步 ainvoke。"""

    def __init__(self, name, *, description="", parameters=None, result=None, exc=None):
        self.name = name
        self.description = description
        self.args_schema = parameters if parameters is not None else {"type": "object"}
        self.result = result
        self.exc = exc
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeClient:
    """模拟 MultiServerMCPClient：按 server 返回工具，可单独让某个 server 失败。"""

    def __init__(self, servers):
        self.connections = {s: {} for s in servers}
        self._tools = {s: [] for s in servers}
        self.fail = set()
        self.requested = []

    def set_tools(self, server, tools):
        self._tools[server] = list(tools)

    def fail_server(self, server):
        self.fail.add(server)

    async def get_tools(self, *, server_name=None):
        self.requested.append(server_name)
        if server_name in self.fail:
            raise TimeoutError(f"discovery timed out for {server_name}")
        return list(self._tools.get(server_name, []))


def build_provider(tmp_path, *, servers=("logistics", "after_sales"),
                   policy_mcp=None, discovery_timeout=1.0):
    from app.tools.mcp_client import MCPToolProvider
    from app.tools.policy import PolicyStore

    policy_path = tmp_path / "permissions.json"
    make_policy(policy_path, policy_mcp or {
        "logistics": {"query_logistics": "read"},
        "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
    })
    client = FakeClient(servers)
    provider = MCPToolProvider(client, PolicyStore(policy_path), discovery_timeout)
    return provider, client, policy_path


# 1. 再次刷新可见新工具，不沿用旧快照

async def test_refresh_sees_new_tool_on_later_refresh(tmp_path):
    provider, client, _ = build_provider(
        tmp_path, policy_mcp={"logistics": {"query_logistics": "read",
                                            "query_estimate": "read"}})
    client.set_tools("logistics", [FakeTool("logistics_query_logistics")])
    first = await provider.refresh()
    assert set(first) == {"logistics_query_logistics"}

    client.set_tools("logistics", [
        FakeTool("logistics_query_logistics"),
        FakeTool("logistics_query_estimate"),
    ])
    second = await provider.refresh()
    assert set(second) == {"logistics_query_logistics", "logistics_query_estimate"}


# 2. 补权限前第二个工具不可见，补权限后热加载出现

async def test_second_tool_hidden_until_authorized_then_appears(tmp_path):
    provider, client, policy_path = build_provider(
        tmp_path, policy_mcp={"logistics": {"query_logistics": "read"}})
    client.set_tools("logistics", [
        FakeTool("logistics_query_logistics"),
        FakeTool("logistics_query_estimate"),
    ])
    first = await provider.refresh()
    assert set(first) == {"logistics_query_logistics"}

    make_policy(policy_path, {"logistics": {"query_logistics": "read",
                                            "query_estimate": "read"}})
    second = await provider.refresh()
    assert set(second) == {"logistics_query_logistics", "logistics_query_estimate"}


# 3. 一个 server 失败/超时，另一个继续；不暴露失败 server 的旧工具

async def test_failing_server_contributes_no_tools_others_continue(tmp_path):
    provider, client, _ = build_provider(tmp_path)
    client.set_tools("logistics", [FakeTool("logistics_query_logistics")])
    client.set_tools("after_sales", [FakeTool("after_sales_query_warranty")])

    healthy = await provider.refresh()
    assert set(healthy) == {"logistics_query_logistics", "after_sales_query_warranty"}

    client.fail_server("logistics")
    result = await provider.refresh()
    assert set(result) == {"after_sales_query_warranty"}
    # 失败 server 的旧工具不得沿用
    assert "logistics_query_logistics" not in result


async def test_server_timeout_treated_as_failure(tmp_path):
    provider, client, _ = build_provider(tmp_path, discovery_timeout=0.05)
    client.set_tools("after_sales", [FakeTool("after_sales_query_warranty")])

    async def slow_get_tools(*, server_name=None):
        await asyncio.sleep(10)

    original = client.get_tools

    async def get_tools(*, server_name=None):
        if server_name == "logistics":
            await asyncio.sleep(10)
        return await original(server_name=server_name)

    client.get_tools = get_tools
    result = await provider.refresh()
    assert set(result) == {"after_sales_query_warranty"}


# 4. 两个 server 同名工具由前缀隔离，original_name 保留

async def test_same_raw_name_prefixed_and_original_preserved(tmp_path):
    provider, client, _ = build_provider(
        tmp_path, policy_mcp={"logistics": {"query_status": "read"},
                              "after_sales": {"query_status": "read"}})
    client.set_tools("logistics", [FakeTool("logistics_query_status")])
    client.set_tools("after_sales", [FakeTool("after_sales_query_status")])

    tools = await provider.refresh()
    assert set(tools) == {"logistics_query_status", "after_sales_query_status"}
    assert tools["logistics_query_status"].server == "logistics"
    assert tools["logistics_query_status"].original_name == "query_status"
    assert tools["after_sales_query_status"].server == "after_sales"
    assert tools["after_sales_query_status"].original_name == "query_status"


# 5. 撤权移除；schema 变化在刷新后反映，不沿用旧 schema

async def test_revoke_removes_and_schema_change_reflected(tmp_path):
    provider, client, policy_path = build_provider(tmp_path)
    client.set_tools("logistics", [FakeTool(
        "logistics_query_logistics",
        parameters={"type": "object",
                    "properties": {"a": {"type": "string"}}})])

    first = await provider.refresh()
    assert first["logistics_query_logistics"].parameters["properties"]["a"]["type"] == "string"

    make_policy(policy_path, {})  # 撤权
    revoked = await provider.refresh()
    assert revoked == {}

    make_policy(policy_path, {"logistics": {"query_logistics": "read"}})
    client.set_tools("logistics", [FakeTool(
        "logistics_query_logistics",
        parameters={"type": "object",
                    "properties": {"b": {"type": "integer"}}})])
    third = await provider.refresh()
    assert "a" not in third["logistics_query_logistics"].parameters["properties"]
    assert third["logistics_query_logistics"].parameters["properties"]["b"]["type"] == "integer"


# 6. 未知前缀工具跳过，不猜测来源

async def test_unknown_prefix_tool_skipped(tmp_path):
    provider, client, _ = build_provider(tmp_path)
    client.set_tools("logistics", [
        FakeTool("logistics_query_logistics"),
        FakeTool("weird_tool"),  # 没有 logistics_ 前缀
    ])
    tools = await provider.refresh()
    assert set(tools) == {"logistics_query_logistics"}


# 7. 结果适配

async def test_dict_success_adapts_to_success(tmp_path):
    provider, client, _ = build_provider(tmp_path)
    client.set_tools("logistics", [FakeTool(
        "logistics_query_logistics",
        result={"tracking_no": "SF1", "status_code": "delivered"})])
    tools = await provider.refresh()
    outcome = await tools["logistics_query_logistics"].invoke({"tracking_no": "SF1"}, None)
    assert outcome.category == "success"
    assert outcome.data == {"tracking_no": "SF1", "status_code": "delivered"}


async def test_json_string_content_is_parsed(tmp_path):
    provider, client, _ = build_provider(tmp_path)
    client.set_tools("logistics", [FakeTool(
        "logistics_query_logistics",
        result=[{"type": "text", "text": json.dumps({"x": 1, "events": []})}])])
    tools = await provider.refresh()
    outcome = await tools["logistics_query_logistics"].invoke({}, None)
    assert outcome.category == "success"
    assert outcome.data == {"x": 1, "events": []}


async def test_tool_exception_is_fault_not_success(tmp_path):
    provider, client, _ = build_provider(tmp_path)
    client.set_tools("logistics", [FakeTool(
        "logistics_query_logistics", exc=ToolException("上游业务失败"))])
    tools = await provider.refresh()
    outcome = await tools["logistics_query_logistics"].invoke({}, None)
    assert outcome.category == "fault"
    assert outcome.data is None
    assert "上游业务失败" in outcome.message


async def test_transient_transport_exception_raises_transient(tmp_path):
    provider, client, _ = build_provider(tmp_path)
    client.set_tools("logistics", [FakeTool(
        "logistics_query_logistics", exc=httpx.ConnectError("connection refused"))])
    tools = await provider.refresh()
    with pytest.raises(TransientToolError):
        await tools["logistics_query_logistics"].invoke({}, None)


async def test_chinese_status_translation_for_known_tools(tmp_path):
    provider, client, _ = build_provider(tmp_path)
    client.set_tools("logistics", [FakeTool(
        "logistics_query_logistics",
        result=[{"type": "text", "text": json.dumps({
            "tracking_no": "SF1", "status_code": "in_transit",
            "events": [{"time": "t1", "status_code": "pending_pickup"},
                       {"time": "t2", "status_code": "delivered"}]})}])])
    client.set_tools("after_sales", [
        FakeTool("after_sales_query_warranty", result={"status_code": "in_warranty"}),
        FakeTool("after_sales_query_return_progress",
                 result={"status_code": "refunded",
                         "progress": [{"time": "t", "status_code": "pending"}]}),
    ])
    tools = await provider.refresh()

    outcome = await tools["logistics_query_logistics"].invoke({}, None)
    assert outcome.data["status_code"] == "in_transit"
    formatted = tools["logistics_query_logistics"].formatter(outcome)
    assert formatted.data["status_code"] == "in_transit"  # 保留 ASCII 码
    assert formatted.data["status_text"] == "运输中"
    assert formatted.data["events"][0]["status_text"] == "待揽收"
    assert formatted.data["events"][1]["status_text"] == "已签收"

    outcome = await tools["after_sales_query_warranty"].invoke({}, None)
    formatted = tools["after_sales_query_warranty"].formatter(outcome)
    assert formatted.data["status_code"] == "in_warranty"
    assert formatted.data["status_text"] == "在保"

    outcome = await tools["after_sales_query_return_progress"].invoke({}, None)
    formatted = tools["after_sales_query_return_progress"].formatter(outcome)
    assert formatted.data["status_code"] == "refunded"
    assert formatted.data["status_text"] == "已退款"
    assert formatted.data["progress"][0]["status_text"] == "待处理"


async def test_unknown_tool_gets_no_invented_translation(tmp_path):
    provider, client, _ = build_provider(
        tmp_path, policy_mcp={"logistics": {"query_estimate": "read"}})
    client.set_tools("logistics", [FakeTool(
        "logistics_query_estimate", result={"status_code": "delivered"})])
    tools = await provider.refresh()
    definition = tools["logistics_query_estimate"]
    assert definition.formatter is None
    outcome = await definition.invoke({}, None)
    assert "status_text" not in outcome.data


# create_mcp_client / 配置字段

def test_create_mcp_client_builds_connections():
    from app.tools.mcp_client import create_mcp_client

    class Settings:
        mcp_logistics_url = "http://127.0.0.1:8101/mcp"
        mcp_after_sales_url = "http://127.0.0.1:8102/mcp"

    client = create_mcp_client(Settings)
    assert client.tool_name_prefix is True
    assert client.handle_tool_errors is False
    assert client.connections["logistics"]["url"] == "http://127.0.0.1:8101/mcp"
    assert client.connections["logistics"]["transport"] in {"streamable_http", "http"}
    assert client.connections["after_sales"]["url"] == "http://127.0.0.1:8102/mcp"
    assert client.connections["after_sales"]["transport"] in {"streamable_http", "http"}


def test_config_fields_added():
    from app.config import Settings

    s = Settings(_env_file=None)
    assert s.mcp_logistics_url == "http://127.0.0.1:8101/mcp"
    assert s.mcp_after_sales_url == "http://127.0.0.1:8102/mcp"
    assert s.tool_permissions_path == "config/tool_permissions.json"
    assert s.mcp_discovery_timeout == 3.0


# 8. 真实 SDK Server 集成：启动两个进程，经 provider 发现并调用三个工具

async def test_real_servers_discover_and_invoke(tmp_path):
    from app.tools.mcp_client import MCPToolProvider, create_mcp_client
    from app.tools.policy import PolicyStore

    port_log = free_port()
    port_after = free_port()

    procs = []
    try:
        procs.append(await asyncio.create_subprocess_exec(
            sys.executable, "-m", "mcp_servers.logistics",
            "--host", "127.0.0.1", "--port", str(port_log),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL))
        procs.append(await asyncio.create_subprocess_exec(
            sys.executable, "-m", "mcp_servers.after_sales",
            "--host", "127.0.0.1", "--port", str(port_after),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL))

        class Settings:
            mcp_logistics_url = f"http://127.0.0.1:{port_log}/mcp"
            mcp_after_sales_url = f"http://127.0.0.1:{port_after}/mcp"

        client = create_mcp_client(Settings)
        policy_path = tmp_path / "permissions.json"
        make_policy(policy_path, {
            "logistics": {"query_logistics": "read"},
            "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
        })
        provider = MCPToolProvider(client, PolicyStore(policy_path), discovery_timeout=3.0)

        # 轮询至两个 server 就绪（有界 ~20s）
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 20
        ready = False
        while loop.time() < deadline:
            try:
                log_tools = await asyncio.wait_for(
                    client.get_tools(server_name="logistics"), timeout=2)
                after_tools = await asyncio.wait_for(
                    client.get_tools(server_name="after_sales"), timeout=2)
                if log_tools and len(after_tools) == 2:
                    ready = True
                    break
            except Exception:
                await asyncio.sleep(0.2)
        assert ready, "两个 MCP server 未在 20s 内就绪"

        tools = await provider.refresh()
        assert set(tools) == {"logistics_query_logistics",
                              "after_sales_query_warranty",
                              "after_sales_query_return_progress"}

        ctx = ExecutionContext(conversation_id=1, tool_call_id="it-1")

        outcome = await tools["logistics_query_logistics"].invoke(
            {"tracking_no": "SF-IT-1"}, ctx)
        assert outcome.category == "success"
        assert outcome.data["tracking_no"] == "SF-IT-1"
        assert outcome.data["mock"] is True
        assert outcome.data["status_code"] in LOGISTICS_CODES
        formatted = tools["logistics_query_logistics"].formatter(outcome)
        assert formatted.data["status_text"] in set(LOGISTICS_TEXT.values())
        assert formatted.data["status_code"] in LOGISTICS_CODES

        outcome = await tools["after_sales_query_warranty"].invoke(
            {"product_name": "耳机"}, ctx)
        assert outcome.category == "success"
        assert outcome.data["product_name"] == "耳机"
        assert outcome.data["mock"] is True
        assert outcome.data["status_code"] in WARRANTY_CODES
        formatted = tools["after_sales_query_warranty"].formatter(outcome)
        assert formatted.data["status_text"] in set(WARRANTY_TEXT.values())

        outcome = await tools["after_sales_query_return_progress"].invoke(
            {"return_no": "R-IT-1"}, ctx)
        assert outcome.category == "success"
        assert outcome.data["return_no"] == "R-IT-1"
        assert outcome.data["mock"] is True
        assert outcome.data["status_code"] in RETURN_CODES
        formatted = tools["after_sales_query_return_progress"].formatter(outcome)
        assert formatted.data["status_text"] in set(RETURN_TEXT.values())
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                await asyncio.wait_for(p.wait(), timeout=5)
            except Exception:
                p.kill()
