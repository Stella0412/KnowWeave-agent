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
import sys
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
from app.prompts.customer_service import SYSTEM_PROMPT
from app.tools.builtin import register_builtins
from app.tools.registry import ToolRegistry
from app.tools.ticket_authorization import update_ticket_request

CASES_PATH = Path(__file__).resolve().parent / "ch08_ticket_cases.json"

SECURITY_CATEGORIES = {"否定", "引用", "条件句"}

# 模型「请用户补充描述」的文本信号（启发式，只用于把模型输出归类，不参与授权）
_ASK_DESCRIPTION_MARKERS = (
    "描述", "请补充", "补充一下", "具体", "详细", "什么问题", "请问",
)


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


async def _run_case(bound, turns: list[str]) -> str:
    """喂真实模型跑一个会话的多轮，返回最后一轮的归类。"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    observed = "no_chat_ticket"
    for turn in turns:
        messages.append(HumanMessage(content=turn))
        ai = await bound.ainvoke(messages)
        observed = _classify(ai)
        # 历史里只保留纯文本回复，避免悬挂 tool_calls 干扰后续轮
        messages.append(AIMessage(content=_text_of(getattr(ai, "content", ""))))
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
            observed = await _run_case(bound, case["turns"])
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["intent"], default="intent")
    args = parser.parse_args()
    if args.mode == "intent":
        return asyncio.run(run_intent())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
