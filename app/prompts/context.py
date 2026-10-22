"""ch07：摘要 Prompt、动态上下文块、证据/订单预算截断。"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.prompts.orchestrator import (
    AGENT_SYSTEM_ADDENDUM_AFTER_SALE,
    AGENT_SYSTEM_ADDENDUM_BUSINESS,
    AGENT_SYSTEM_ADDENDUM_CLARIFY,
    AGENT_SYSTEM_ADDENDUM_KNOWLEDGE,
    AGENT_SYSTEM_ADDENDUM_LOGISTICS,
    AGENT_SYSTEM_ADDENDUM_REFUND,
)
from app.services.token_counter import TokenCounter


SUMMARY_SYSTEM = """你是电商客服会话的摘要器。只提炼事实与诉求，不回答用户、不续写对话。

必须保留（对话里出现才写）：
- 用户问过/买过的商品或品类；
- 订单号、手机号、物流单号；
- 用户明确提出的诉求（退款、换货、催发货、补发等）；
- 尚未解决的问题。

禁止：
- 对话里没有出现的内容一个字不编；不确定就不写。
- 不写寒暄、感谢、客服客套话。
- 不写「用户咨询了相关问题」这类空话。

长度：目标 50–200 个汉字。事实极少时可以短于 50，但不得为空；不得为凑字数加寒暄。
超长时优先保留订单号、商品、诉求、未解决问题。
表述尽量精简：用最少的字概括同一事实，删去「了」「的」「吗」等不影响语义的虚词。"""


class SummaryOutput(BaseModel):
    has_facts: bool = Field(description="这段对话是否包含可提炼的业务事实或诉求")
    summary: str = Field(default="", description="50–200 字事实摘要；没有事实时为空")


def summary_messages(transcript: str) -> list:
    return [
        SystemMessage(content=SUMMARY_SYSTEM),
        HumanMessage(content=f"【对话转写】\n{transcript}"),
    ]


def _route_addendum(route: str, intent: str | None) -> str:
    if route == "knowledge":
        return AGENT_SYSTEM_ADDENDUM_KNOWLEDGE
    if route == "clarify":
        return AGENT_SYSTEM_ADDENDUM_CLARIFY
    if route == "refund_flow":
        return (
            AGENT_SYSTEM_ADDENDUM_REFUND
            if intent == "refund_return"
            else AGENT_SYSTEM_ADDENDUM_AFTER_SALE
        )
    if intent == "logistics":
        return AGENT_SYSTEM_ADDENDUM_LOGISTICS
    if route == "business":
        return AGENT_SYSTEM_ADDENDUM_BUSINESS
    return ""


def truncate_evidence(text: str, *, max_tokens: int, counter: TokenCounter) -> str:
    text = text or ""
    if not text or counter.count_text(text) <= max_tokens:
        return text
    marker = "\n…（证据已截断）"
    parts = [part for part in re.split(r"(?=\[\d+\])", text) if part]
    kept: list[str] = []
    for part in parts:
        if counter.count_text("".join(kept + [part]) + marker) > max_tokens:
            break
        kept.append(part)
    if kept:
        return "".join(kept).rstrip() + marker
    return counter.truncate_text(text, max_tokens, marker)


_ORDER_FIELDS = [
    "order_no", "product_name", "product_type", "status",
    "signed_at", "amount", "tracking_no", "paid_at",
]


def compact_order_json(order: dict, *, max_tokens: int, counter: TokenCounter) -> str:
    data = {key: order[key] for key in _ORDER_FIELDS if key in (order or {})}
    dropped = False
    while data:
        text = json.dumps(data, ensure_ascii=False, default=str)
        if counter.count_text(text) <= max_tokens:
            break
        removed = data.popitem()[0]
        dropped = True
        if removed == "__truncated__":
            break
    if dropped:
        data["__truncated__"] = True
    return json.dumps(data, ensure_ascii=False, default=str)


def build_dynamic_block(
    *,
    route: str,
    intent: str | None,
    summary: str,
    evidence: str | None,
    order_data: dict | None,
    counter: TokenCounter,
    evidence_budget_tokens: int,
    order_max_tokens: int,
) -> str:
    addendum = _route_addendum(route, intent)
    sections: list[str] = []
    if summary:
        sections.append(f"【历史梗概】\n{summary}")
    if route in {"knowledge", "refund_flow"}:
        clipped = truncate_evidence(
            evidence or "", max_tokens=evidence_budget_tokens, counter=counter
        )
        sections.append(f"【检索证据】\n{clipped or '（无）'}")
    elif evidence:
        clipped = truncate_evidence(
            evidence, max_tokens=evidence_budget_tokens, counter=counter
        )
        sections.append(f"【检索证据】\n{clipped}")
    if addendum:
        sections.append(f"【本轮路由】\n{addendum}")
    if order_data:
        sections.append(
            "【本轮订单】\n"
            + compact_order_json(order_data, max_tokens=order_max_tokens, counter=counter)
        )
    if order_data or route == "refund_flow":
        sections.append(
            "【判定优先级】本轮订单/证据是唯一判定依据；历史订单与旧结论只作背景。"
        )
    if not sections:
        return ""
    return "【本轮上下文资料｜仅本轮有效】\n" + "\n\n".join(sections)
