from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.extract import AfterSaleExtraction

EXTRACT_SYSTEM_PROMPT = (
    "你是售后信息提取助手。从用户的售后描述中提取订单号、诉求类型、期望方案。"
    "诉求类型只能是：退货退款、换货、仅退款、维修、物流查询、其他。"
    "用户没提供订单号时 order_id 返回 null。"
)


class ExtractService:
    def __init__(self, structured_model):
        self._model = structured_model

    async def extract(self, text: str) -> AfterSaleExtraction:
        messages = [
            SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=text),
        ]
        return await self._model.ainvoke(messages)
