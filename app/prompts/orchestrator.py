"""ch05 编排层 Prompt：意图分类、固定话术、Agent 路由补充指令。"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

IntentName = Literal[
    "logistics", "order", "product_consult", "refund_return",
    "after_sale", "complaint", "chitchat", "other",
]


class IntentOutput(BaseModel):
    intent: IntentName = Field(description="用户这一条消息的唯一意图类别")
    confidence: float = Field(
        ge=0.0, le=1.0, description="对 intent 判断的置信度，0~1"
    )


INTENT_SYSTEM = """你是电商客服的意图分类器。只输出 intent 和 confidence 两个字段，不要解释。

八类定义：
- logistics：查物流进度、快递到哪、派送、签收、承运商、物流单号。
- order：查订单状态、金额、商品、下单/取消/修改订单、发货时间；不涉及退款退货政策。
- product_consult：商品价格、库存、规格、材质、适用对象、优惠、保质期等售前咨询。
- refund_return：退货、退款、换货、运费谁承担、退款到账、退货条件等退款退货政策。
- after_sale：商品破损、少件、漏发、质量问题、保修、发票、赠品补发等事实售后；不含退款退货政策。
- complaint：投诉、维权、态度差、欺骗、要说法、要求领导处理、明确要求转人工投诉。
- chitchat：问候、感谢、闲聊、问你是谁、天气、笑话、告别。
- other：超出电商客服范围、多意图冲突无法判断、没有明确业务动作、拿不准。

边界 few-shot：
- 「拆封了还能退吗」→ refund_return（问政策）
- 「收到的猫粮少了一袋」→ after_sale（陈述事实售后）
- 「订单 1001 到哪了」→ logistics
- 「我要投诉物流」→ complaint
- 「你们总部在哪」→ other
- 「那个东西怎么回事」→ other（没有明确业务动作）
- 「帮我写首诗」→ other

confidence 口径：明确 0.9+；边界 0.6~0.8；拿不准 <0.6 并优先 other。
禁止把超出范围或拿不准的问题硬塞进七类业务意图。"""


def intent_messages(question: str, history=None) -> list:
    history_text = _history_text(history) if history else ""
    prefix = f"【最近对话】\n{history_text}\n\n" if history_text else ""
    return [
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=f"{prefix}【当前用户消息】\n{question}"),
    ]


class CorefOutput(BaseModel):
    resolved_query: str = Field(
        description="补全指代和省略后的完整问题；无需改写时原样返回用户原话"
    )


COREF_SYSTEM = """你是电商客服的指代消解器。只输出 resolved_query 一个字段。

规则：
1. 结合最近对话，把「它能退吗」「那明天呢」这类带指代/省略的话补全成脱离上下文也能看懂的完整问句。
2. 只补指代和省略，把口语归一成标准问法；不回答、不解释、不新增诉求。
3. 订单号、型号、数字、时限、商品名一律保留，不改写、不补全。
4. 原问题已经完整、指代已明确 → 原样返回，不强行改写。
5. 无法判断指代指什么 → 原样返回，不猜。"""


def _history_text(history) -> str:
    lines = []
    for message in history or []:
        role = "用户" if isinstance(message, HumanMessage) else "客服"
        text = message.content if isinstance(message.content, str) else str(message.content or "")
        if text.strip():
            lines.append(f"{role}：{text.strip()}")
    return "\n".join(lines)


def coref_messages(history, current: str) -> list:
    history_text = _history_text(history) or "（无历史）"
    return [
        SystemMessage(content=COREF_SYSTEM),
        HumanMessage(
            content=f"【最近对话】\n{history_text}\n\n【当前用户消息】\n{current}"
        ),
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
