"""ch08 · 真实联调集成测试（Task 11）。

覆盖：真实 langchain-mcp-adapters 双 Server 发现/调用、仅存在于临时 Server 源码的新读工具
在授权前不可见、只重启该 Server + 更新临时权限文件后客服进程 PID 不变且下一轮发现、
真实模型轨迹包含新公开名、真实 HTTP 工单确认/取消（MySQL）、两独立进程并发防重、
用户 DDL 属性（unsigned / 中文 ENUM / 无外键 / utf8mb4）。

门控：需要 CH08_LIVE=1（真实 LLM + MySQL 容器 knowweave-ch08-mysql@127.0.0.1:3307 + 本地
MCP 子进程）。默认 `pytest -q` 保持 hermetic，门控未开时显式跳过并打印原因，绝不静默。
"""

import asyncio
import json
import os
import re
import socket
import sys
import uuid
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
MYSQL_URL = "mysql+aiomysql://root:knowweave123@127.0.0.1:3307/knowweave?charset=utf8mb4"

LIVE = os.environ.get("CH08_LIVE") == "1"
_SKIP_REASON = (
    "需要 CH08_LIVE=1：真实 LLM + MySQL 容器 knowweave-ch08-mysql(127.0.0.1:3307) + 本地 MCP 子进程；"
    "默认套件保持 hermetic 故跳过。运行方式：CH08_LIVE=1 .venv/Scripts/python.exe -m pytest tests/test_ch08_integration.py -q"
)
live = pytest.mark.skipif(not LIVE, reason=_SKIP_REASON)

if not LIVE:
    print(f"[ch08-integration] 门控未开（CH08_LIVE!=1）：{_SKIP_REASON}", file=sys.stderr)


# —— 仅存在于临时 Server 源码的物流 Server（新增读工具 query_estimate，绝不入 git 的 mcp_servers/）——

TEMP_LOGISTICS_SERVER = '''"""Temporary logistics MCP server for ch08 acceptance (NOT committed)."""

import argparse

from mcp.server.fastmcp import FastMCP

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8101


def query_logistics(tracking_no: str) -> dict:
    """按运单号查询物流轨迹（mock 数据）。"""
    return {
        "tracking_no": tracking_no,
        "carrier": "顺丰速运",
        "status_code": "in_transit",
        "events": [
            {"time": "2026-10-20T08:00:00+00:00", "status_code": "pending_pickup"},
            {"time": "2026-10-21T08:00:00+00:00", "status_code": "in_transit"},
        ],
        "mock": True,
    }


def query_estimate(tracking_no: str) -> dict:
    """查询运单预计送达时间。用户询问什么时候送到、还要多久、预计送达时间时调用本工具。"""
    return {
        "tracking_no": tracking_no,
        "eta": "预计 2026-10-26 送达",
        "days_left": 2,
        "mock": True,
    }


def create_server(host, port, *, with_estimate=False):
    mcp = FastMCP("logistics", host=host, port=port, stateless_http=True, json_response=True)
    mcp.tool()(query_logistics)
    if with_estimate:
        mcp.tool()(query_estimate)
    return mcp


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--with-estimate", action="store_true")
    args = parser.parse_args(argv)
    create_server(host=args.host, port=args.port, with_estimate=args.with_estimate).run(
        transport="streamable-http"
    )


if __name__ == "__main__":
    main()
'''


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def write_policy(path: Path, mcp: dict) -> Path:
    path.write_text(json.dumps({"mcp": mcp}, ensure_ascii=False), encoding="utf-8")
    return path


def write_temp_logistics_server(tmp_path: Path) -> Path:
    path = tmp_path / "tmp_logistics.py"
    path.write_text(TEMP_LOGISTICS_SERVER, encoding="utf-8")
    return path


async def terminate_procs(procs: list) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            await asyncio.wait_for(p.wait(), timeout=5)
        except Exception:
            p.kill()


async def spawn_mcp_servers(tmp_path: Path, *, logistics_extra: bool = False):
    """启动两个独立 MCP 子进程；logistics 使用临时 Server 源码（可带新工具）。"""
    logistics_src = write_temp_logistics_server(tmp_path)
    log_port = free_port()
    after_port = free_port()
    procs = []

    logistics_cmd = [sys.executable, str(logistics_src), "--host", "127.0.0.1",
                     "--port", str(log_port)]
    if logistics_extra:
        logistics_cmd.append("--with-estimate")
    procs.append(await asyncio.create_subprocess_exec(
        *logistics_cmd, cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL))
    procs.append(await asyncio.create_subprocess_exec(
        sys.executable, "-m", "mcp_servers.after_sales",
        "--host", "127.0.0.1", "--port", str(after_port),
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL))
    return procs, log_port, after_port


async def wait_servers_ready(client, *, expect_logistics: int, expect_after: int, deadline: float = 25.0):
    from app.tools.mcp_client import MCPToolProvider

    loop = asyncio.get_running_loop()
    end = loop.time() + deadline
    log_tools = []
    while loop.time() < end:
        try:
            log_tools = await asyncio.wait_for(client.get_tools(server_name="logistics"), timeout=2)
            after_tools = await asyncio.wait_for(client.get_tools(server_name="after_sales"), timeout=2)
            if len(log_tools) == expect_logistics and len(after_tools) == expect_after:
                return log_tools, after_tools
        except Exception:
            await asyncio.sleep(0.2)
    raise AssertionError(
        f"MCP servers 未在 {deadline}s 内就绪 (logistics={len(log_tools)})"
    )


async def spawn_app(tmp_path: Path, *, mcp_logistics_url: str, mcp_after_sales_url: str,
                    permissions_path: Path):
    port = free_port()
    log_file = tmp_path / "app.log"
    checkpoint = tmp_path / "checkpoints.sqlite"
    env = {
        **os.environ,
        "MYSQL_HOST": "127.0.0.1",
        "MYSQL_PORT": "3307",
        "MYSQL_USER": "root",
        "MYSQL_PASSWORD": "knowweave123",
        "MYSQL_DB": "knowweave",
        "MCP_LOGISTICS_URL": mcp_logistics_url,
        "MCP_AFTER_SALES_URL": mcp_after_sales_url,
        "TOOL_PERMISSIONS_PATH": str(permissions_path),
        "LOG_FILE": str(log_file),
        "LANGGRAPH_CHECKPOINT_PATH": str(checkpoint),
    }
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", str(port),
        cwd=str(REPO_ROOT), env=env,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    return {"proc": proc, "port": port, "log_file": log_file}


async def wait_app_ready(port: int, deadline: float = 60.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + deadline
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=2.0) as client:
        while loop.time() < end:
            try:
                resp = await client.get("/")
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.3)
    raise AssertionError(f"app 未在 {deadline}s 内就绪（端口 {port}）")


async def chat_sse(client: httpx.AsyncClient, message: str, session_id: str | None = None):
    """POST /chat 并逐条产出解析后的 SSE 事件 dict。"""
    body = {"message": message}
    if session_id is not None:
        body["session_id"] = session_id
    async with client.stream("POST", "/chat", json=body, timeout=120.0) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            yield json.loads(line[len("data:"):].strip())


async def chat_events(client: httpx.AsyncClient, message: str, session_id: str | None = None):
    return [event async for event in chat_sse(client, message, session_id)]


async def resume_ticket(client: httpx.AsyncClient, session_id: str, confirmation_id: str,
                        decision: str):
    async with client.stream(
        "POST", "/chat/ticket/resume",
        json={"session_id": session_id, "confirmation_id": confirmation_id, "decision": decision},
        timeout=120.0,
    ) as resp:
        body = await resp.aread()
        return resp.status_code, body.decode("utf-8", errors="replace")


def _event_tool_names(events: list) -> list[str]:
    names = []
    for event in events:
        if event.get("type") == "tool_status":
            names.append(event.get("name", ""))
    return names


def _ticket_preview(events: list) -> dict | None:
    for event in events:
        if event.get("type") != "actions":
            continue
        for item in event.get("items") or []:
            if item.get("type") == "ticket_preview":
                return item
    return None


def _session_id(events: list) -> str | None:
    for event in events:
        if event.get("type") == "session":
            return str(event.get("session_id"))
    return None


async def mysql_count(engine, table: str, *, where: str = "") -> int:
    stmt = f"SELECT COUNT(*) FROM {table}"
    if where:
        stmt += f" WHERE {where}"
    async with engine.connect() as conn:
        return int((await conn.execute(text(stmt))).scalar())


# 真实模型（非零温度）对建单触发语偶发改走「查订单 + 澄清」，用有限次新会话重试
# 打开确认中断；每次都开新会话，绝不污染前一次的中断状态。
_TICKET_TRIGGERS = (
    "帮我建个工单，反馈订单 1001 发货太慢",
    "帮我建个工单，订单 1001 发货太慢，我要跟进",
    "帮我建个工单，订单 1001 我要修改收货地址",
)


async def open_ticket_interrupt(client, max_attempts: int = 6):
    last_events: list = []
    for i in range(max_attempts):
        message = _TICKET_TRIGGERS[i % len(_TICKET_TRIGGERS)]
        events = await chat_events(client, message)
        last_events = events
        cid = _session_id(events)
        preview = _ticket_preview(events)
        if cid is not None and preview is not None:
            return events, cid, preview
    raise AssertionError(
        f"{max_attempts} 次尝试均未打开工单确认中断（真实模型触发语不稳定）: {last_events}"
    )


# —— 1. 真实 adapters 双 Server 发现并调用三个工具 ——

@live
@pytest.mark.asyncio
async def test_real_client_discovers_and_invokes_both_servers(tmp_path):
    from app.tools.contracts import ExecutionContext
    from app.tools.mcp_client import MCPToolProvider, create_mcp_client
    from app.tools.policy import PolicyStore

    procs, log_port, after_port = await spawn_mcp_servers(tmp_path)
    try:
        class Settings:
            mcp_logistics_url = f"http://127.0.0.1:{log_port}/mcp"
            mcp_after_sales_url = f"http://127.0.0.1:{after_port}/mcp"

        client = create_mcp_client(Settings)
        policy_path = write_policy(tmp_path / "permissions.json", {
            "logistics": {"query_logistics": "read"},
            "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
        })
        provider = MCPToolProvider(client, PolicyStore(policy_path), discovery_timeout=3.0)
        await wait_servers_ready(client, expect_logistics=1, expect_after=2)

        tools = await provider.refresh()
        assert set(tools) == {
            "logistics_query_logistics",
            "after_sales_query_warranty",
            "after_sales_query_return_progress",
        }

        ctx = ExecutionContext(conversation_id=1, tool_call_id="live-1")
        outcome = await tools["logistics_query_logistics"].invoke({"tracking_no": "SF-LIVE-1"}, ctx)
        assert outcome.category == "success" and outcome.data["tracking_no"] == "SF-LIVE-1"

        outcome = await tools["after_sales_query_warranty"].invoke({"product_name": "耳机"}, ctx)
        assert outcome.category == "success" and outcome.data["product_name"] == "耳机"

        outcome = await tools["after_sales_query_return_progress"].invoke({"return_no": "R-LIVE-1"}, ctx)
        assert outcome.category == "success" and outcome.data["return_no"] == "R-LIVE-1"
    finally:
        await terminate_procs(procs)


# —— 2. 新工具授权前不可见；重启该 Server + 更新权限文件后才出现 ——

@live
@pytest.mark.asyncio
async def test_hot_tool_invisible_until_restart_and_authorize(tmp_path):
    from app.tools.mcp_client import MCPToolProvider, create_mcp_client
    from app.tools.policy import PolicyStore

    procs, log_port, after_port = await spawn_mcp_servers(tmp_path, logistics_extra=False)
    try:
        class Settings:
            mcp_logistics_url = f"http://127.0.0.1:{log_port}/mcp"
            mcp_after_sales_url = f"http://127.0.0.1:{after_port}/mcp"

        client = create_mcp_client(Settings)
        policy_path = tmp_path / "permissions.json"
        write_policy(policy_path, {
            "logistics": {"query_logistics": "read"},
            "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
        })
        provider = MCPToolProvider(client, PolicyStore(policy_path), discovery_timeout=3.0)
        await wait_servers_ready(client, expect_logistics=1, expect_after=2)

        first = await provider.refresh()
        assert set(first) == {
            "logistics_query_logistics",
            "after_sales_query_warranty",
            "after_sales_query_return_progress",
        }
        assert "logistics_query_estimate" not in first

        # 重启仅 logistics 一个 Server：换成带新工具的临时源码；权限文件暂不授权新工具。
        procs[0].terminate()
        await asyncio.wait_for(procs[0].wait(), timeout=5)
        logistics_src = tmp_path / "tmp_logistics.py"
        procs[0] = await asyncio.create_subprocess_exec(
            sys.executable, str(logistics_src), "--host", "127.0.0.1", "--port", str(log_port),
            "--with-estimate", cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await wait_servers_ready(client, expect_logistics=2, expect_after=2)

        second = await provider.refresh()
        # 已注册但未授权 → 依然不可见
        assert "logistics_query_estimate" not in second

        # 更新权限文件（热加载，无需重建 provider）
        write_policy(policy_path, {
            "logistics": {"query_logistics": "read", "query_estimate": "read"},
            "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
        })
        third = await provider.refresh()
        assert "logistics_query_estimate" in third
        assert third["logistics_query_estimate"].server == "logistics"
        assert third["logistics_query_estimate"].original_name == "query_estimate"
    finally:
        await terminate_procs(procs)


# —— 3. 运行中的 app（PID 不变）下一轮发现新工具，且真实模型轨迹包含新公开名 ——

@live
@pytest.mark.asyncio
async def test_app_hot_reloads_new_tool_next_turn_with_model_trajectory(tmp_path):
    procs, log_port, after_port = await spawn_mcp_servers(tmp_path, logistics_extra=False)
    permissions_path = tmp_path / "permissions.json"
    write_policy(permissions_path, {
        "logistics": {"query_logistics": "read"},
        "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
    })
    app = None
    engine = None
    try:
        app = await spawn_app(
            tmp_path,
            mcp_logistics_url=f"http://127.0.0.1:{log_port}/mcp",
            mcp_after_sales_url=f"http://127.0.0.1:{after_port}/mcp",
            permissions_path=permissions_path,
        )
        app_pid = app["proc"].pid
        await wait_app_ready(app["port"])

        engine = create_async_engine(MYSQL_URL)
        base_url = f"http://127.0.0.1:{app['port']}"
        question = "帮我查一下运单 SF-HOT-001 大概什么时候能送到？"

        # 共享 MySQL 会残留历史审计行：用增量而不是绝对值做隔离断言。
        estimate_baseline = await mysql_count(
            engine, "tool_audit_logs", where="tool_name='logistics_query_estimate'")

        async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
            before_events = await chat_events(client, question)
        before_tools = _event_tool_names(before_events)
        assert "logistics_query_estimate" not in before_tools, before_tools
        assert await mysql_count(engine, "tool_audit_logs",
                                 where="tool_name='logistics_query_estimate'") == estimate_baseline

        # 重启仅 logistics Server（换成带新工具），并热更新权限文件；app 进程不动。
        procs[0].terminate()
        await asyncio.wait_for(procs[0].wait(), timeout=5)
        logistics_src = tmp_path / "tmp_logistics.py"
        procs[0] = await asyncio.create_subprocess_exec(
            sys.executable, str(logistics_src), "--host", "127.0.0.1", "--port", str(log_port),
            "--with-estimate", cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        write_policy(permissions_path, {
            "logistics": {"query_logistics": "read", "query_estimate": "read"},
            "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
        })

        async with engine.connect() as conn:
            max_id = (await conn.execute(text(
                "SELECT COALESCE(MAX(id), 0) FROM tool_audit_logs"))).scalar()

        async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
            after_events = await chat_events(client, question)
        after_tools = _event_tool_names(after_events)
        assert "logistics_query_estimate" in after_tools, after_tools
        assert app["proc"].pid == app_pid

        # 真实轨迹落库：本轮新产生了新公开名的成功审计（mcp_server=logistics）。
        async with engine.connect() as conn:
            rows = (await conn.execute(text(
                "SELECT tool_name, mcp_server, status FROM tool_audit_logs "
                "WHERE tool_name='logistics_query_estimate' AND id > :max_id"
            ), {"max_id": max_id})).fetchall()
        assert rows, "新工具没有产生审计记录"
        assert all(r[2] == "成功" for r in rows), rows
        assert all(r[1] == "logistics" for r in rows), rows
    finally:
        if app is not None:
            app["proc"].terminate()
            try:
                await asyncio.wait_for(app["proc"].wait(), timeout=5)
            except Exception:
                app["proc"].kill()
        await terminate_procs(procs)
        if engine is not None:
            await engine.dispose()


# —— 4. 真实 HTTP 工单确认/取消 + MySQL（stale ID / 重复恢复拒绝，无额外写入） ——

@live
@pytest.mark.asyncio
async def test_ticket_confirm_cancel_over_http_and_mysql(tmp_path):
    procs, log_port, after_port = await spawn_mcp_servers(tmp_path)
    permissions_path = tmp_path / "permissions.json"
    write_policy(permissions_path, {
        "logistics": {"query_logistics": "read"},
        "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
    })
    app = None
    engine = None
    try:
        app = await spawn_app(
            tmp_path,
            mcp_logistics_url=f"http://127.0.0.1:{log_port}/mcp",
            mcp_after_sales_url=f"http://127.0.0.1:{after_port}/mcp",
            permissions_path=permissions_path,
        )
        await wait_app_ready(app["port"])
        engine = create_async_engine(MYSQL_URL)
        base_url = f"http://127.0.0.1:{app['port']}"

        async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
            # 确认链路会话（真实模型触发语偶发不稳，带有限次新会话重试）
            events, cid, preview = await open_ticket_interrupt(client)
            confirmation_id = preview["confirmation_id"]

            # 历史接口返回 pending_action
            hist = await client.get(f"/api/conversations/{cid}/messages")
            assert hist.status_code == 200
            pending = hist.json()["pending_action"]
            assert pending is not None and pending["confirmation_id"] == confirmation_id

            tickets_before = await mysql_count(engine, "tickets")
            audits_before = await mysql_count(engine, "tool_audit_logs")

            # stale confirmation_id → 409，无写入
            status, _ = await resume_ticket(client, cid, "TC" + "0" * 30, "confirm")
            assert status == 409
            assert await mysql_count(engine, "tickets") == tickets_before
            assert await mysql_count(engine, "tool_audit_logs") == audits_before

            # 确认 → 恰好 1 张工单 + 1 条成功审计（create_ticket / builtin）
            success_audits_before = await mysql_count(
                engine, "tool_audit_logs",
                where="tool_name='create_ticket' AND status='成功'")
            status, _ = await resume_ticket(client, cid, confirmation_id, "confirm")
            assert status == 200
            assert await mysql_count(engine, "tickets") == tickets_before + 1
            assert await mysql_count(
                engine, "tool_audit_logs",
                where="tool_name='create_ticket' AND status='成功'") == success_audits_before + 1
            async with engine.connect() as conn:
                audit = (await conn.execute(text(
                    "SELECT status, tool_name, tool_source FROM tool_audit_logs "
                    "WHERE tool_name='create_ticket' AND status='成功' ORDER BY id DESC LIMIT 1"
                ))).fetchone()
                ticket_no = (await conn.execute(text(
                    "SELECT ticket_no FROM tickets ORDER BY created_at DESC, ticket_no DESC LIMIT 1"
                ))).scalar()
            assert audit[0] == "成功" and audit[1] == "create_ticket" and audit[2] == "builtin"
            assert ticket_no and re.fullmatch(r"T8[0-9a-f]{30}", ticket_no)

            # 重复恢复（同一 confirmation_id）→ 409，无额外写入
            status, _ = await resume_ticket(client, cid, confirmation_id, "confirm")
            assert status == 409
            assert await mysql_count(engine, "tickets") == tickets_before + 1
            assert await mysql_count(
                engine, "tool_audit_logs",
                where="tool_name='create_ticket' AND status='成功'") == success_audits_before + 1

            # 取消链路：新会话 → 0 工单 + 权限拒绝审计（error_message 含「取消」）
            events, cid2, preview2 = await open_ticket_interrupt(client)
            confirmation_id2 = preview2["confirmation_id"]
            tickets_before2 = await mysql_count(engine, "tickets")
            denied_before2 = await mysql_count(
                engine, "tool_audit_logs", where="status='权限拒绝'")

            status, _ = await resume_ticket(client, cid2, confirmation_id2, "cancel")
            assert status == 200
            assert await mysql_count(engine, "tickets") == tickets_before2
            denied_after2 = await mysql_count(
                engine, "tool_audit_logs", where="status='权限拒绝'")
            assert denied_after2 >= denied_before2 + 1
            async with engine.connect() as conn:
                audit = (await conn.execute(text(
                    "SELECT status, error_message FROM tool_audit_logs "
                    "WHERE status='权限拒绝' ORDER BY id DESC LIMIT 1"
                ))).fetchone()
            assert audit[0] == "权限拒绝" and "取消" in (audit[1] or "")
    finally:
        if app is not None:
            app["proc"].terminate()
            try:
                await asyncio.wait_for(app["proc"].wait(), timeout=5)
            except Exception:
                app["proc"].kill()
        await terminate_procs(procs)
        if engine is not None:
            await engine.dispose()


# —— 5. 两个独立进程并发争抢同一稳定工单号，只产生一行 ——

_RACE_WORKER = '''\
import sys, asyncio

sys.path.insert(0, {repo_root!r})

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from app.tools.ticket_confirmation import TicketDraft, create_ticket_once


async def main():
    conv_id = int(sys.argv[1])
    url = sys.argv[2]
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    draft = TicketDraft.create(
        conversation_id=conv_id, tool_call_id="race-call-1",
        ticket_type="售后", description="并发防重", request_message_id=1,
    )
    ticket_no = await create_ticket_once(factory, draft, escalate=False)
    print(ticket_no, flush=True)
    await engine.dispose()


asyncio.run(main())
'''


@live
@pytest.mark.asyncio
async def test_two_processes_race_same_ticket_number_one_row(tmp_path):
    engine = None
    workers = []
    try:
        engine = create_async_engine(MYSQL_URL)
        from app.db.base import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        race_uid = "race-" + uuid.uuid4().hex[:8]
        async with engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO conversations (user_id, status) VALUES (:u, '进行中')"
            ), {"u": race_uid})
            conv_id = (await conn.execute(text(
                "SELECT id FROM conversations WHERE user_id = :u"
            ), {"u": race_uid})).scalar()

        worker_src = _RACE_WORKER.format(repo_root=str(REPO_ROOT))
        worker_path = tmp_path / "race_worker.py"
        worker_path.write_text(worker_src, encoding="utf-8")

        for _ in range(2):
            workers.append(await asyncio.create_subprocess_exec(
                sys.executable, str(worker_path), str(conv_id), MYSQL_URL,
                cwd=str(REPO_ROOT),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE))

        outputs = []
        for p in workers:
            out, err = await asyncio.wait_for(p.communicate(), timeout=60)
            outputs.append((p.returncode, out.decode().strip(), err.decode().strip()))
        for code, out, err in outputs:
            assert code == 0, (out, err)
        assert outputs[0][1] == outputs[1][1]
        ticket_no = outputs[0][1]

        async with engine.connect() as conn:
            count = (await conn.execute(text(
                "SELECT COUNT(*) FROM tickets WHERE ticket_no = :no"
            ), {"no": ticket_no})).scalar()
        assert count == 1
    finally:
        if engine is not None:
            await engine.dispose()


# —— 6. 用户 DDL 属性：unsigned 列、中文 ENUM、无外键、utf8mb4 ——

@live
@pytest.mark.asyncio
async def test_tool_audit_ddl_properties_on_mysql():
    engine = create_async_engine(MYSQL_URL)
    try:
        async with engine.connect() as conn:
            cols = (await conn.execute(text(
                "SELECT COLUMN_NAME, COLUMN_TYPE FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA='knowweave' AND TABLE_NAME='tool_audit_logs' "
                "ORDER BY ORDINAL_POSITION"
            ))).fetchall()
        col_types = {c[0]: c[1] for c in cols}
        assert col_types["id"].startswith("bigint unsigned"), col_types["id"]
        assert col_types["conversation_id"].startswith("bigint unsigned")
        assert col_types["retry_count"].startswith("tinyint unsigned")
        assert col_types["duration_ms"].startswith("int unsigned")

        status_type = col_types["status"]
        for value in ("成功", "失败", "超时", "校验拦下", "权限拒绝"):
            assert value in status_type, status_type

        async with engine.connect() as conn:
            fk = (await conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.KEY_COLUMN_USAGE "
                "WHERE TABLE_SCHEMA='knowweave' AND TABLE_NAME='tool_audit_logs' "
                "AND REFERENCED_TABLE_NAME IS NOT NULL"
            ))).scalar()
            table = (await conn.execute(text(
                "SELECT ENGINE, TABLE_COLLATION FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA='knowweave' AND TABLE_NAME='tool_audit_logs'"
            ))).fetchone()
        assert fk == 0
        assert table[0] == "InnoDB"
        assert table[1].startswith("utf8mb4")
    finally:
        await engine.dispose()
