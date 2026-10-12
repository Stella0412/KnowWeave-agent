import asyncio
import logging
import random
from typing import Literal

from langchain_core.tools import BaseTool, tool

from app.config import settings
from app.core.retrieval import NOT_FOUND, UNAVAILABLE, format_hits, search_knowledge
from app.db.models import ConversationStatus
from app.db.repository import ChatRepository

logger = logging.getLogger(__name__)


@tool
async def query_order(order_id: str) -> str:
    """查询订单信息。当用户询问订单状态、订单内容时调用。"""
    statuses = ["待付款", "已付款", "已发货", "已签收"]
    return (
        f"订单 {order_id}：状态={random.choice(statuses)}，"
        f"金额=¥{random.randint(20, 5000)}，商品=知序精选商品"
    )


@tool
async def query_product(product_name: str) -> str:
    """查询商品信息。当用户询问商品价格、多少钱、库存、规格时调用。"""
    return (
        f"商品「{product_name}」：价格=¥{random.randint(9, 999)}，"
        f"库存={random.randint(0, 500)}件，规格=标准款"
    )


@tool
async def query_logistics(order_id: str) -> str:
    """查询物流信息。当用户询问物流到哪了、什么时候送达时调用。"""
    carriers = ["中通", "圆通", "顺丰", "韵达"]
    nodes = ["已揽收", "运输中", "到达转运中心", "派送中"]
    return (
        f"订单 {order_id} 物流：承运商={random.choice(carriers)}，"
        f"当前节点={random.choice(nodes)}，预计3日内送达"
    )


def build_query_faq(session_factory) -> BaseTool:
    """入参/出参契约不变：模型只填 keyword，返回可直接念给用户的文本。

    内脏已从「关键词 LIKE 查 faq 表」换成「向量语义检索 Milvus knowledge 集合」。
    session_factory 参数保留以不动 registry.build_tools，RAG 路径下不再使用。
    """

    @tool
    async def query_faq(keyword: str) -> str:
        """查询常见问题。当用户询问退货/换货/发货/邮费/运费等政策类问题时调用。"""
        try:
            hits = await asyncio.to_thread(search_knowledge, keyword, settings.retrieval_top_k)
            if not hits:
                return NOT_FOUND
            return format_hits(hits)
        except Exception:
            logger.exception("知识库检索失败")
            return UNAVAILABLE

    return query_faq


def build_create_ticket(session_factory, conversation_id: int) -> BaseTool:
    @tool
    async def create_ticket(
        description: str, ticket_type: Literal["售后", "投诉", "咨询"]
    ) -> str:
        """创建人工工单。当用户投诉、维权、或要求转人工等需要人工介入时调用。"""
        async with session_factory() as session:
            repo = ChatRepository(session)
            ticket_no = await repo.create_ticket(conversation_id, description, ticket_type)
            await repo.set_conversation_status(conversation_id, ConversationStatus.ESCALATED)
            await session.commit()
        return f"已为您创建工单，工单号 {ticket_no}，人工客服会尽快处理。"

    return create_ticket
