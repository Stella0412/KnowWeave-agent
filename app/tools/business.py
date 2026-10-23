"""ch08 业务工具的数据访问与检索助手。

内置工具的注册与 execute 适配迁移到 app/tools/builtin.py；这里保留仍被
routers、脚本与直接单测使用的数据访问助手：订单查询、FAQ 检索、建单落库。
query_logistics 已移交 MCP Server，query_product / submit_refund 已整体迁入
builtin.py，不再以 @tool 形式暴露。
"""

import asyncio
import logging
from typing import Literal

from langchain_core.tools import BaseTool, tool

from app.core.retrieval import UNAVAILABLE
from app.db.models import ConversationStatus
from app.db.repository import ChatRepository, LowConfidenceRepository
from app.rag.pipeline import RagResult, TurnContext, default_source_files, run_pipeline
from app.rag.refusal import pool_record, refusal_text

logger = logging.getLogger(__name__)


REFUND_REASON_CATEGORIES = ("七天无理由", "质量问题", "发错货", "不想要了", "其他")


def build_query_order(session_factory) -> BaseTool:
    from app.db.repository import OrderRepository

    @tool
    async def query_order(order_id: str) -> str:
        """查询订单信息。当用户询问订单状态、订单内容、签收时间时调用。"""
        async with session_factory() as session:
            order = await OrderRepository(session).get(order_id)
        if order is None:
            return f"未找到订单 {order_id}，请核对订单号。"
        signed = order.signed_at.strftime("%Y-%m-%d") if order.signed_at else "未签收"
        return (
            f"订单 {order.order_no}：状态={order.status}，商品={order.product_name}，"
            f"类型={order.product_type}，金额=¥{order.amount}，"
            f"物流单号={order.tracking_no or '无'}，签收时间={signed}"
        )

    return query_order


def build_query_faq(session_factory, turn: TurnContext | None = None) -> BaseTool:
    """入参/出参契约不变：模型只填 keyword，返回可直接念给用户的文本。

    内脏从 ch03 的 dense 单路换成 app/rag 的完整管线（混合检索 + 精排 + 引用编号 + 两段拒答）。
    拒答时返回带用户问题的拒答话术而不是空串：模型拿到一句明确的「暂时没查到」才会照实说，
    拿到空串它反而容易用自己的常识补全。
    """

    @tool
    async def query_faq(keyword: str) -> str:
        """查询常见问题。当用户询问退货/换货/发货/邮费/运费等政策类问题时调用。"""
        # 检索有意用模型填的 keyword，而不是 turn.raw_question：turn 可能是 None（脚本/单测直调
        # 工具），那时只有 keyword 可用；用户原话只用于落低置信度池（_record_low_confidence）。
        try:
            result: RagResult = await asyncio.to_thread(
                run_pipeline,
                keyword,
                source_files=(turn.source_files if turn is not None else default_source_files()),
            )
        except Exception:
            logger.exception("知识库检索失败")
            return UNAVAILABLE

        if result.refused:
            await _record_low_confidence(session_factory, turn, result)
            raw_question = turn.raw_question if turn is not None else ""
            return refusal_text(raw_question or keyword)

        if turn is not None:
            turn.citations.extend(result.citations)
        return result.evidence_text

    return query_faq


async def _record_low_confidence(session_factory, turn: TurnContext | None, result: RagResult) -> None:
    """落低置信度问题池。**失败只记 warning，不影响本轮回答**（Review Focus 6）。

    池子写的是**用户原话**（`turn.raw_question`）：DDL 的语义就是原话，且改写本身可能把问题带偏，
    拿改写后的问法落池会掩盖这一点。
    """
    if session_factory is None:
        return
    raw_question = turn.raw_question if turn is not None else result.raw_query
    conversation_id = turn.conversation_id if turn is not None else None
    record = pool_record(raw_question, result.refusal_source, result.refusal_reason)
    try:
        async with session_factory() as session:
            await LowConfidenceRepository(session).add(conversation_id, **record)
            await session.commit()
    except Exception:
        logger.warning("低置信度问题落池失败（不影响本轮回答）", exc_info=True)


async def create_ticket_record(
    session_factory,
    conversation_id: int,
    description: str,
    ticket_type,
    *,
    escalate: bool = True,
) -> str:
    async with session_factory() as session:
        repo = ChatRepository(session)
        ticket_no = await repo.create_ticket(conversation_id, description, ticket_type)
        if escalate:
            await repo.set_conversation_status(conversation_id, ConversationStatus.ESCALATED)
        await session.commit()
    return ticket_no


def build_create_ticket(
    session_factory, conversation_id: int, *, escalate: bool = True
) -> BaseTool:
    @tool
    async def create_ticket(
        description: str, ticket_type: Literal["售后", "投诉", "咨询"]
    ) -> str:
        """创建人工工单。当用户投诉、维权、或要求转人工等需要人工介入时调用。"""
        ticket_no = await create_ticket_record(
            session_factory, conversation_id, description, ticket_type, escalate=escalate
        )
        return f"已为您创建工单，工单号 {ticket_no}，人工客服会尽快处理。"

    return create_ticket
