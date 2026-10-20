"""ch05 编排层 Prompt：意图分类、固定话术、Agent 路由补充指令。"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

IntentName = Literal[
    "logistics", "order", "product_consult", "refund_return",
    "after_sale", "complaint", "chitchat",
]


class IntentOutput(BaseModel):
    intent: IntentName = Field(description="用户这一条消息的唯一意图类别")


INTENT_SYSTEM = """你是电商客服的意图分类器。只输出一个 intent 字段，不要解释。

七类定义：
- logistics：查物流进度、快递到哪、派送、签收、承运商、物流单号。
- order：查订单状态、金额、商品、下单/取消/修改订单、发货时间；不涉及退款退货政策。
- product_consult：商品价格、库存、规格、材质、适用对象、优惠、保质期等售前咨询。
- refund_return：退货、退款、换货、运费谁承担、退款到账、退货条件等政策。
- after_sale：商品破损、少件、漏发、质量问题、保修、发票、赠品补发等售后处理；不含退款退货政策。
- complaint：投诉、维权、态度差、欺骗、要说法、要求领导处理、明确要求转人工投诉。
- chitchat：问候、感谢、闲聊、问你是谁、天气、笑话、告别。

判定优先级：明确说「投诉/维权/要说法」→ complaint；问退款退货政策 → refund_return；
问订单/物流具体信息 → order/logistics；商品售前 → product_consult；其他售后 → after_sale；
没有业务诉求的寒暄 → chitchat。只选一个最贴近的类别。"""


def intent_messages(question: str) -> list:
    return [
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=f"用户消息：{question}"),
    ]
