"""ch08 · 即插即用工具系统演示 CLI（Task 11）。

五种模式：
- mcp           启动自己两个独立 MCP Server，用官方 langchain-mcp-adapters 发现并调用，
                打印 PID / 公开工具名 / call_id / 耗时。
- hot-reload    物流 Server 先不带新工具，发现后「只重启该 Server + 更新临时权限文件」，
                再发现一次，展示新公开名出现（不经过任何公开注册端点）。
- ticket-confirm 对 --base-url 指向的运行中 app 走真实 HTTP 确认链路，打印
                HTTP 状态 / session_id / confirmation_id / MySQL 里的工单行与审计行。
- ticket-cancel 同上，走取消链路，打印 权限拒绝 审计行、确认零工单写入。
- timeouts      启动一个受控慢 MCP 读工具，用短超时触发超时分类（绝不写真实工单）。

只启动/清理本脚本自己创建的辅助进程；所有权限/配置走临时文件 + 独立端口，
不覆盖用户当前会话与权限。base-url 仅在 ticket-* / timeouts 模式使用（指向运行中 app）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import time
import uuid
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
MYSQL_URL = "mysql+aiomysql://root:knowweave123@127.0.0.1:3307/knowweave?charset=utf8mb4"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 仅存在于临时 Server 源码的物流 Server（带可开关的新读工具 query_estimate，不入库）
TEMP_LOGISTICS_SERVER = '''\
"""Temporary logistics MCP server for ch08 demo (NOT committed)."""

import argparse

from mcp.server.fastmcp import FastMCP


def query_logistics(tracking_no: str) -> dict:
    return {
        "tracking_no": tracking_no,
        "carrier": "顺丰速运",
        "status_code": "in_transit",
        "events": [{"time": "2026-10-21T08:00:00+00:00", "status_code": "in_transit"}],
        "mock": True,
    }


def query_estimate(tracking_no: str) -> dict:
    """查询运单预计送达时间。用户询问什么时候送到、还要多久、预计送达时间时调用本工具。"""
    return {"tracking_no": tracking_no, "eta": "预计 2026-10-26 送达", "days_left": 2, "mock": True}


def create_server(host, port, *, with_estimate=False):
    mcp = FastMCP("logistics", host=host, port=port, stateless_http=True, json_response=True)
    mcp.tool()(query_logistics)
    if with_estimate:
        mcp.tool()(query_estimate)
    return mcp


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--with-estimate", action="store_true")
    args = parser.parse_args(argv)
    create_server(host=args.host, port=args.port, with_estimate=args.with_estimate).run(
        transport="streamable-http"
    )


if __name__ == "__main__":
    main()
'''

# 受控慢 MCP 读工具：只用于演示超时分类，绝不写真实工单
SLOW_SERVER = '''\
"""Temporary slow MCP server for ch08 timeout demo (NOT committed)."""

import argparse
import time

from mcp.server.fastmcp import FastMCP


def slow_read(tracking_no: str, seconds: float = 5.0) -> dict:
    """受控慢读工具：按秒数睡眠后返回，仅用于演示超时。"""
    time.sleep(min(max(seconds, 0.0), 30.0))
    return {"tracking_no": tracking_no, "slow": True}


def create_server(host, port):
    mcp = FastMCP("slow", host=host, port=port, stateless_http=True, json_response=True)
    mcp.tool()(slow_read)
    return mcp


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8103)
    args = parser.parse_args(argv)
    create_server(host=args.host, port=args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
'''


def _emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def write_temp(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def write_policy(path: Path, mcp: dict) -> Path:
    path.write_text(json.dumps({"mcp": mcp}, ensure_ascii=False), encoding="utf-8")
    return path


async def _terminate(procs: list) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            await asyncio.wait_for(p.wait(), timeout=5)
        except Exception:
            p.kill()


async def _spawn_server(cmd: list, label: str) -> asyncio.subprocess.Process:
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    _emit("server_started", server=label, pid=proc.pid)
    return proc


async def _spawn_temp_server(src: Path, port: int, label: str, extra: list | None = None):
    return await _spawn_server(
        [sys.executable, str(src), "--host", "127.0.0.1", "--port", str(port), *(extra or [])],
        label,
    )


async def _wait_ready(client, *, expect_logistics: int | None = None,
                      expect_after: int | None = None, expect_slow: int | None = None,
                      deadline: float = 25.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + deadline
    while loop.time() < end:
        try:
            if expect_logistics is not None:
                log = await asyncio.wait_for(client.get_tools(server_name="logistics"), timeout=2)
                if len(log) != expect_logistics:
                    raise AssertionError(f"logistics={len(log)}")
            if expect_after is not None:
                after = await asyncio.wait_for(client.get_tools(server_name="after_sales"), timeout=2)
                if len(after) != expect_after:
                    raise AssertionError(f"after_sales={len(after)}")
            if expect_slow is not None:
                slow = await asyncio.wait_for(client.get_tools(server_name="slow"), timeout=2)
                if len(slow) != expect_slow:
                    raise AssertionError(f"slow={len(slow)}")
            return
        except Exception:
            await asyncio.sleep(0.2)
    raise SystemExit(f"ERROR: MCP servers 未在 {deadline}s 内就绪")


async def run_mcp() -> int:
    from app.tools.contracts import ExecutionContext
    from app.tools.mcp_client import MCPToolProvider, create_mcp_client
    from app.tools.policy import PolicyStore

    tmp = REPO_ROOT / "data" / "demo_ch08"
    tmp.mkdir(parents=True, exist_ok=True)
    logistics_src = write_temp(tmp / "tmp_logistics.py", TEMP_LOGISTICS_SERVER)
    log_port, after_port = free_port(), free_port()
    procs = [
        await _spawn_temp_server(logistics_src, log_port, "logistics"),
        await _spawn_server(
            [sys.executable, "-m", "mcp_servers.after_sales",
             "--host", "127.0.0.1", "--port", str(after_port)],
            "after_sales",
        ),
    ]
    try:
        class Settings:
            mcp_logistics_url = f"http://127.0.0.1:{log_port}/mcp"
            mcp_after_sales_url = f"http://127.0.0.1:{after_port}/mcp"

        client = create_mcp_client(Settings)
        policy_path = write_policy(tmp / "permissions.json", {
            "logistics": {"query_logistics": "read"},
            "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
        })
        provider = MCPToolProvider(client, PolicyStore(policy_path), discovery_timeout=3.0)
        await _wait_ready(client, expect_logistics=1, expect_after=2)
        tools = await provider.refresh()
        _emit("tools_discovered", tools=sorted(tools))

        ctx = ExecutionContext(conversation_id=0, tool_call_id=f"demo-{uuid.uuid4().hex[:8]}")
        for public_name, args in (
            ("logistics_query_logistics", {"tracking_no": "SF-DEMO-1"}),
            ("after_sales_query_warranty", {"product_name": "耳机"}),
            ("after_sales_query_return_progress", {"return_no": "R-DEMO-1"}),
        ):
            started = time.perf_counter()
            outcome = await tools[public_name].invoke(args, ctx)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _emit("tool_invoked", public_name=public_name, call_id=ctx.tool_call_id,
                  category=outcome.category, elapsed_ms=elapsed_ms,
                  data=outcome.data if outcome.data is not None else None)
        return 0
    finally:
        await _terminate(procs)


async def run_hot_reload() -> int:
    from app.tools.mcp_client import MCPToolProvider, create_mcp_client
    from app.tools.policy import PolicyStore

    tmp = REPO_ROOT / "data" / "demo_ch08"
    tmp.mkdir(parents=True, exist_ok=True)
    logistics_src = write_temp(tmp / "tmp_logistics.py", TEMP_LOGISTICS_SERVER)
    log_port, after_port = free_port(), free_port()
    procs = [
        await _spawn_temp_server(logistics_src, log_port, "logistics"),
        await _spawn_server(
            [sys.executable, "-m", "mcp_servers.after_sales",
             "--host", "127.0.0.1", "--port", str(after_port)],
            "after_sales",
        ),
    ]
    try:
        class Settings:
            mcp_logistics_url = f"http://127.0.0.1:{log_port}/mcp"
            mcp_after_sales_url = f"http://127.0.0.1:{after_port}/mcp"

        client = create_mcp_client(Settings)
        policy_path = tmp / "permissions.json"
        write_policy(policy_path, {
            "logistics": {"query_logistics": "read"},
            "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
        })
        provider = MCPToolProvider(client, PolicyStore(policy_path), discovery_timeout=3.0)
        await _wait_ready(client, expect_logistics=1, expect_after=2)
        before = await provider.refresh()
        _emit("hot_reload_before", tools=sorted(before),
              new_visible="logistics_query_estimate" in before)

        # 只重启 logistics 一个 Server（换成带新工具的临时源码），权限文件先不授权新工具
        procs[0].terminate()
        await asyncio.wait_for(procs[0].wait(), timeout=5)
        procs[0] = await _spawn_temp_server(
            logistics_src, log_port, "logistics", extra=["--with-estimate"])
        await _wait_ready(client, expect_logistics=2, expect_after=2)
        after_restart = await provider.refresh()
        _emit("hot_reload_after_restart_unauthed", tools=sorted(after_restart),
              new_visible="logistics_query_estimate" in after_restart)

        # 热更新权限文件（不重建 provider），再发现：新公开名出现
        write_policy(policy_path, {
            "logistics": {"query_logistics": "read", "query_estimate": "read"},
            "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
        })
        after_auth = await provider.refresh()
        _emit("hot_reload_after_auth", tools=sorted(after_auth),
              new_visible="logistics_query_estimate" in after_auth,
              new_public_name="logistics_query_estimate" if "logistics_query_estimate" in after_auth else None)
        return 0 if "logistics_query_estimate" in after_auth else 1
    finally:
        await _terminate(procs)


async def _app_chat_sse(base_url: str, message: str) -> list[dict]:
    events = []
    async with httpx.AsyncClient(base_url=base_url, timeout=180.0) as client:
        async with client.stream("POST", "/chat", json={"message": message}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    events.append(json.loads(line[len("data:"):].strip()))
    return events


def _first(events: list[dict], kind: str) -> dict | None:
    for e in events:
        if e.get("type") == kind:
            return e
    return None


def _ticket_confirmation_id(events: list[dict]) -> str | None:
    for e in events:
        if e.get("type") != "actions":
            continue
        for item in e.get("items") or []:
            if item.get("type") == "ticket_preview":
                return item.get("confirmation_id")
    return None


async def _resume(base_url: str, session_id: str, confirmation_id: str, decision: str):
    async with httpx.AsyncClient(base_url=base_url, timeout=180.0) as client:
        async with client.stream(
            "POST", "/chat/ticket/resume",
            json={"session_id": session_id, "confirmation_id": confirmation_id,
                  "decision": decision},
        ) as resp:
            body = await resp.aread()
            return resp.status_code, body.decode("utf-8", errors="replace")


async def _mysql_rows():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(MYSQL_URL)
    try:
        async with engine.connect() as conn:
            ticket = (await conn.execute(text(
                "SELECT ticket_no, ticket_type, status FROM tickets "
                "ORDER BY created_at DESC, ticket_no DESC LIMIT 1"
            ))).mappings().first()
            audits = (await conn.execute(text(
                "SELECT tool_name, tool_source, status, error_message FROM tool_audit_logs "
                "ORDER BY id DESC LIMIT 3"
            ))).mappings().all()
        return dict(ticket) if ticket else None, [dict(a) for a in audits]
    finally:
        await engine.dispose()


TICKET_MESSAGE = "帮我建个工单，反馈订单 1001 发货太慢"


async def _run_ticket(base_url: str, decision: str) -> int:
    events = await _app_chat_sse(base_url, TICKET_MESSAGE)
    session_id = None
    session_event = _first(events, "session")
    if session_event is not None:
        session_id = str(session_event.get("session_id"))
    confirmation_id = _ticket_confirmation_id(events)
    if not session_id or not confirmation_id:
        _emit("ticket_interrupt", session_id=session_id, confirmation_id=confirmation_id,
              events=events[-6:])
        print("ERROR: 未拿到 ticket_preview 中断（session_id/confirmation_id 缺失）")
        return 1

    status, body = await _resume(base_url, session_id, confirmation_id, decision)
    ticket_row, audit_rows = await _mysql_rows()
    _emit("ticket_resume", decision=decision, session_id=session_id,
          confirmation_id=confirmation_id, http_status=status,
          ticket_row=ticket_row, audit_rows=audit_rows,
          sse_tail=body[-400:])
    if status != 200:
        print(f"ERROR: /chat/ticket/resume 返回 {status}")
        return 1
    return 0


async def run_ticket(base_url: str, decision: str) -> int:
    # 先确认 app 可达（base-url 由用户指向已启动的 app）
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        try:
            await client.get("/")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: app 不可达（{base_url}）: {exc}")
            return 2
    return await _run_ticket(base_url, decision)


async def run_timeouts() -> int:
    from app.tools.contracts import ExecutionContext
    from app.tools.mcp_client import MCPToolProvider, create_mcp_client
    from app.tools.policy import PolicyStore

    tmp = REPO_ROOT / "data" / "demo_ch08"
    tmp.mkdir(parents=True, exist_ok=True)
    slow_src = write_temp(tmp / "tmp_slow.py", SLOW_SERVER)
    slow_port = free_port()
    procs = [await _spawn_temp_server(slow_src, slow_port, "slow")]
    try:
        class Settings:
            mcp_logistics_url = "http://127.0.0.1:1/mcp"  # 占位，不会被用到
            mcp_after_sales_url = "http://127.0.0.1:1/mcp"

        client = create_mcp_client(Settings)
        client.connections["slow"] = {"url": f"http://127.0.0.1:{slow_port}/mcp",
                                      "transport": "streamable_http"}
        policy_path = write_policy(tmp / "permissions_slow.json", {"slow": {"slow_read": "read"}})
        provider = MCPToolProvider(client, PolicyStore(policy_path), discovery_timeout=3.0)
        await _wait_ready(client, expect_slow=1)
        tools = await provider.refresh()
        definition = tools["slow_slow_read"]
        ctx = ExecutionContext(conversation_id=0, tool_call_id=f"demo-{uuid.uuid4().hex[:8]}")

        started = time.perf_counter()
        try:
            await asyncio.wait_for(
                definition.invoke({"tracking_no": "SLOW-1", "seconds": 5.0}, ctx),
                timeout=1.0,
            )
            _emit("timeout_result", public_name="slow_slow_read", timed_out=False,
                  elapsed_ms=int((time.perf_counter() - started) * 1000))
            return 1  # 没超时反而不符合演示预期
        except asyncio.TimeoutError:
            _emit("timeout_result", public_name="slow_slow_read", timed_out=True,
                  elapsed_ms=int((time.perf_counter() - started) * 1000),
                  classification="超时")
            return 0
    finally:
        await _terminate(procs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=["mcp", "hot-reload", "ticket-confirm", "ticket-cancel", "timeouts"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    if args.mode == "mcp":
        return asyncio.run(run_mcp())
    if args.mode == "hot-reload":
        return asyncio.run(run_hot_reload())
    if args.mode == "ticket-confirm":
        return asyncio.run(run_ticket(args.base_url, "confirm"))
    if args.mode == "ticket-cancel":
        return asyncio.run(run_ticket(args.base_url, "cancel"))
    if args.mode == "timeouts":
        return asyncio.run(run_timeouts())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
