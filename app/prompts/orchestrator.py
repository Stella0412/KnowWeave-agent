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


COMPLAINT_REPLY = (
    "非常抱歉给您带来了不好的体验，您反馈的问题我已经记录下来了。"
    "您可以选择转接人工客服，或者让我先为您创建一个工单，后续由专人跟进处理。"
)

CHITCHAT_REPLY = (
    "您好呀～我是知序的客服知序，很高兴为您服务。"
    "我主要可以帮您处理商品咨询、订单、物流和售后问题，有其他需要也可以告诉我哦。"
)

KNOWLEDGE_UNAVAILABLE_REPLY = (
    "抱歉，知识库暂时不可用，请稍后再试；您也可以联系人工客服核实。"
)

AGENT_STOPPED_REPLY = (
    "抱歉，这个问题我暂时没能完整处理，建议您联系人工客服进一步核实。"
)

AGENT_SYSTEM_ADDENDUM_KNOWLEDGE = """【本轮是知识类问题】
- 下面已经给出本轮强制检索到的证据；不要再调用 query_faq。
- 只依据证据回答，引用编号沿用 [n] 规则；证据没有的数字、条件、时限不要补。
- 如果还需要订单/物流等业务数据，可以调用 query_order / query_logistics / query_product。
- 需要人工介入时调用 create_ticket；它只会生成一个前端按钮提议，不会立刻建单。"""

AGENT_SYSTEM_ADDENDUM_BUSINESS = """【本轮是业务数据类问题】
- 订单、物流、商品信息必须调用对应工具查询后再回答，不要凭记忆编造。
- 查询物流前，如果只有订单号，先调用 query_order 拿到物流单号，再调用 query_logistics。
- 需要人工介入时调用 create_ticket；它只会生成一个前端按钮提议，不会立刻建单。
- 普通咨询不要建议转人工。"""
