"""ch08 · 建单意图评估（真实模型 + Prompt/状态 harness）。

--mode intent：用真实配置模型（settings/.env，绝不打印密钥）逐例跑标注集，
只创建测试会话、不写实际工单（create_ticket 在 Task 9 前本就是拦截成预览，
本评估器也不走写库路径）。逐例打印 id / expected / observed / pass，最后给出命中率。

安全用例（否定 / 引用 / 条件句）绝不允许产生预览；一旦出现会在结果中显式点名。
若真实模型无法运行（凭证缺失/不可用、网络失败），直接打印 BLOCKED 并退出，
绝不替换假模型、绝不编造命中率。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.db.models  # noqa: F401  注册表结构，Base.metadata.create_all 才有表可建
from app.config import settings
from app.core.llm import build_chat_model
from app.db.base import Base
from app.db.models import Ticket
from app.db.repository import ChatRepository
from app.prompts.context import _route_addendum, ticket_request_addendum
from app.prompts.customer_service import SYSTEM_PROMPT
from app.tools.builtin import register_builtins
from app.tools.registry import ToolRegistry
from app.tools.ticket_authorization import update_ticket_request

CASES_PATH = Path(__file__).resolve().parent / "ch08_ticket_cases.json"

SECURITY_CATEGORIES = {"否定", "引用", "条件句", "撤回", "仅投诉"}

# 模型「请用户补充描述」的文本信号（启发式，只用于把模型输出归类，不参与授权）
_ASK_DESCRIPTION_MARKERS = ("描述", "补充", "详细", "什么问题", "什么情况")


def _addendum_block(route: str, intent: str | None) -> str:
    """复用生产的路由 addendum 选择器，避免在评估器里复制 Prompt 字符串。"""
    addendum = _route_addendum(route, intent)
    if not addendum:
        return ""
    return f"【本轮路由】\n{addendum}"


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")


def _classify(ai: AIMessage) -> str:
    """把模型输出归类为 preview / ask_description / no_chat_ticket。"""
    for call in getattr(ai, "tool_calls", None) or []:
        if call.get("name") == "create_ticket":
            return "preview"
    text = _text_of(getattr(ai, "content", ""))
    if any(marker in text for marker in _ASK_DESCRIPTION_MARKERS):
        return "ask_description"
    return "no_chat_ticket"


def _blocked_reason() -> str | None:
    if not (settings.llm_api_key or "").strip():
        return "LLM_API_KEY 未配置（settings/.env 缺失或为空），无法运行真实模型"
    return None


async def _run_case(bound, case: dict) -> str:
    """喂真实模型跑一个会话的多轮，返回最后一轮的归类。

    与生产一致：只有「当前轮」携带该 case 路由对应的 addendum（system prompt 之外），
    历史轮保留用户原话、不重复注入 addendum；建单证据按轮用 update_ticket_request
    推导（消息 id 用 1 起的合成序号，只取 status），active 时经与生产相同的
    ticket_request_addendum 追加提示，镜像生产跨轮补全。
    """
    route = case.get("route") or "business"
    intent = case.get("intent")
    block = _addendum_block(route, intent)
    history: list = [SystemMessage(content=SYSTEM_PROMPT)]
    observed = "no_chat_ticket"
    previous_evidence = None
    for index, turn in enumerate(case["turns"], start=1):
        evidence = update_ticket_request(previous_evidence, index, turn)
        ticket_hint = ticket_request_addendum(evidence)
        messages = list(history)
        content = turn + ("\n\n" + block if block else "")
        if ticket_hint:
            content += "\n\n" + ticket_hint
        messages.append(HumanMessage(content=content))
        ai = await bound.ainvoke(messages)
        observed = _classify(ai)
        # 历史按生产语义存原始话术（不含 addendum），只保留纯文本回复
        history.append(HumanMessage(content=turn))
        history.append(AIMessage(content=_text_of(getattr(ai, "content", ""))))
        previous_evidence = evidence
    return observed


async def run_intent() -> int:
    reason = _blocked_reason()
    if reason is not None:
        print(f"BLOCKED: {reason}")
        return 2

    try:
        model = build_chat_model(settings, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKED: 模型初始化失败: {exc}")
        return 2

    registry = ToolRegistry()
    register_builtins(registry)
    tools = [
        {"name": d.name, "description": d.description, "parameters": d.parameters}
        for d in registry.snapshot().values()
    ]
    bound = model.bind_tools(tools)

    create_desc = next(
        (d["description"] for d in tools if d["name"] == "create_ticket"), ""
    )
    print(f"绑定 create_ticket 描述: {create_desc}")

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # 阶段 1：只创建测试会话、落库用户消息，并用落库消息 id/文本推导确定性证据。
    prepared = []
    async with session_factory() as session:
        repo = ChatRepository(session)
        for case in cases:
            cid = await repo.create_conversation()
            previous = None
            final_evidence = None
            for turn in case["turns"]:
                row = await repo.add_message(cid, role="user", content=turn)
                await session.commit()
                evidence = update_ticket_request(previous, row.id, turn)
                previous = evidence
                final_evidence = (
                    evidence.to_dict() if evidence is not None else None
                )
            prepared.append((case, final_evidence))

    # 阶段 2：真实模型逐例跑（不持有 DB 会话）。
    results = []
    security_violations = []
    for case, final_evidence in prepared:
        try:
            observed = await _run_case(bound, case)
        except Exception as exc:  # noqa: BLE001
            await engine.dispose()
            print(f"BLOCKED: 真实模型调用失败（{case['id']}）: {exc}")
            return 2
        passed = observed == case["expected"]
        results.append((case, observed, passed, final_evidence))
        mark = "PASS" if passed else "FAIL"
        print(
            f"[{case['id']}] 期望={case['expected']} 实际={observed} {mark} "
            f"证据={final_evidence['status'] if final_evidence else None}"
        )
        if case["category"] in SECURITY_CATEGORIES and observed == "preview":
            security_violations.append(case["id"])

    # 只创建测试会话，不写实际工单：评估全程零工单行才通过
    async with session_factory() as session:
        ticket_count = (await session.execute(
            select(func.count()).select_from(Ticket)
        )).scalar()
    await engine.dispose()
    if ticket_count:
        print(f"FAIL: 评估过程写入了 {ticket_count} 条工单，预期 0")
        return 1

    hit = sum(1 for _, _, passed, _ in results if passed)
    total = len(results)
    print(f"\n命中率: {hit}/{total} = {hit / total:.1%}" if total else "\n无用例")
    if security_violations:
        print(
            "安全用例违反：以下用例产生了预览，绝不允许："
            + ", ".join(security_violations)
        )
    else:
        print("安全用例（否定/引用/条件句）均未产生预览")
    return 0


# —— trajectory 模式：真实模型 MCP 工具轨迹（双 Server + 热新增工具） ——
#
# 用真实模型 + 官方 langchain-mcp-adapters 跑出工具轨迹，逐例报告；模型不调用预期工具
# 即 FAIL，绝不替换假模型、绝不编造/改标轨迹。热新增工具只在「重启该 Server + 更新
# 临时权限文件」之后才对模型可见。

_TEMP_LOGISTICS_SERVER = '''\
"""Temporary logistics MCP server for ch08 trajectory eval (NOT committed)."""

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


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _wait_trajectory_ready(client, expect_log: int, expect_after: int,
                                 deadline: float = 25.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + deadline
    while loop.time() < end:
        try:
            log = await asyncio.wait_for(client.get_tools(server_name="logistics"), timeout=2)
            after = await asyncio.wait_for(client.get_tools(server_name="after_sales"), timeout=2)
            if len(log) == expect_log and len(after) == expect_after:
                return
        except Exception:
            pass
        await asyncio.sleep(0.2)
    raise RuntimeError(f"MCP servers 未在 {deadline}s 内就绪")


def _tool_dicts(definitions: dict) -> list[dict]:
    return [
        {"name": d.name, "description": d.description, "parameters": d.parameters}
        for d in definitions.values()
    ]


async def _trajectory_of(bound, message: str) -> list[str]:
    ai = await bound.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message),
    ])
    return [c.get("name", "") for c in getattr(ai, "tool_calls", None) or []]


async def run_trajectory() -> int:
    reason = _blocked_reason()
    if reason is not None:
        print(f"BLOCKED: {reason}")
        return 2
    try:
        model = build_chat_model(settings, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKED: 模型初始化失败: {exc}")
        return 2

    from app.tools.mcp_client import MCPToolProvider, create_mcp_client
    from app.tools.policy import PolicyStore

    repo_root = Path(__file__).resolve().parent.parent
    tmp = Path(tempfile.mkdtemp(prefix="ch08-trajectory-"))
    logistics_src = tmp / "tmp_logistics.py"
    logistics_src.write_text(_TEMP_LOGISTICS_SERVER, encoding="utf-8")
    log_port, after_port = _free_port(), _free_port()
    procs = []

    def _write_policy(mcp: dict) -> None:
        (tmp / "permissions.json").write_text(
            json.dumps({"mcp": mcp}, ensure_ascii=False), encoding="utf-8")

    try:
        procs.append(await asyncio.create_subprocess_exec(
            sys.executable, str(logistics_src), "--host", "127.0.0.1",
            "--port", str(log_port), cwd=str(repo_root),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL))
        procs.append(await asyncio.create_subprocess_exec(
            sys.executable, "-m", "mcp_servers.after_sales",
            "--host", "127.0.0.1", "--port", str(after_port), cwd=str(repo_root),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL))

        class _S:
            mcp_logistics_url = f"http://127.0.0.1:{log_port}/mcp"
            mcp_after_sales_url = f"http://127.0.0.1:{after_port}/mcp"

        client = create_mcp_client(_S)
        _write_policy({
            "logistics": {"query_logistics": "read"},
            "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
        })
        provider = MCPToolProvider(
            client, PolicyStore(tmp / "permissions.json"), discovery_timeout=3.0)
        await _wait_trajectory_ready(client, 1, 2)
        definitions = await provider.refresh()
        bound = model.bind_tools(_tool_dicts(definitions))

        cases = [
            ("logistics_tracking", "帮我查一下运单 SF-TRJ-1 的物流轨迹",
             "logistics_query_logistics"),
            ("after_sales_warranty", "帮我查一下耳机这个商品的保修状态，看看还在不在保修期内",
             "after_sales_query_warranty"),
            ("after_sales_return", "我的退货单 R-TRJ-1 现在到哪一步了？",
             "after_sales_query_return_progress"),
            ("pre_hot_estimate", "帮我查一下运单 SF-TRJ-2 大概什么时候能送到？",
             "logistics_query_estimate"),
        ]

        results: list[bool] = []
        for case_id, message, expected in cases:
            try:
                trajectory = await _trajectory_of(bound, message)
            except Exception as exc:  # noqa: BLE001
                print(f"[{case_id}] FAIL: 模型调用异常: {exc}")
                results.append(False)
                continue
            passed = (expected not in trajectory) if case_id == "pre_hot_estimate" \
                else (expected in trajectory)
            results.append(passed)
            print(f"[{case_id}] 轨迹={trajectory} 关键公开名={expected} "
                  f"{'PASS' if passed else 'FAIL'}")

        # 只重启 logistics Server（换成带新工具的源码）+ 更新临时权限文件，不重建 provider
        procs[0].terminate()
        await asyncio.wait_for(procs[0].wait(), timeout=5)
        procs[0] = await asyncio.create_subprocess_exec(
            sys.executable, str(logistics_src), "--host", "127.0.0.1",
            "--port", str(log_port), "--with-estimate", cwd=str(repo_root),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        _write_policy({
            "logistics": {"query_logistics": "read", "query_estimate": "read"},
            "after_sales": {"query_warranty": "read", "query_return_progress": "read"},
        })
        await _wait_trajectory_ready(client, 2, 2)
        definitions = await provider.refresh()
        bound = model.bind_tools(_tool_dicts(definitions))
        try:
            trajectory = await _trajectory_of(
                bound, "帮我查一下运单 SF-TRJ-3 大概什么时候能送到？")
        except Exception as exc:  # noqa: BLE001
            print(f"[hot_estimate] FAIL: 模型调用异常: {exc}")
            trajectory = []
        passed = "logistics_query_estimate" in trajectory
        results.append(passed)
        print(f"[hot_estimate] 轨迹={trajectory} 关键公开名=logistics_query_estimate "
              f"{'PASS' if passed else 'FAIL'}")
        print(f"热新增后可见公开名集合: {sorted(definitions)}")

        hit = sum(1 for p in results if p)
        total = len(results)
        print(f"\n轨迹命中: {hit}/{total}")
        return 0 if hit == total else 1
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                await asyncio.wait_for(p.wait(), timeout=5)
            except Exception:
                p.kill()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["intent", "trajectory"], default="intent")
    args = parser.parse_args()
    if args.mode == "intent":
        return asyncio.run(run_intent())
    if args.mode == "trajectory":
        return asyncio.run(run_trajectory())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
