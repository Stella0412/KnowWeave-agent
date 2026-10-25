"""ch05 编排层 Prompt：意图分类、固定话术、Agent 路由补充指令。"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

IntentName = Literal[
    "logistics", "order", "product_consult", "refund_return",
    "after_sale", "complaint", "human", "chitchat", "other",
]


class IntentOutput(BaseModel):
    intent: IntentName = Field(description="用户这一条消息的唯一意图类别")
    confidence: float = Field(
        ge=0.0, le=1.0, description="对 intent 判断的置信度，0~1"
    )


INTENT_SYSTEM = """你是电商客服的意图分类器。只输出 intent 和 confidence 两个字段，不要解释。

九类定义：
- logistics：查物流进度、快递到哪、派送、签收、承运商、物流单号。
- order：查订单状态、金额、商品、下单/取消/修改订单、发货时间；不涉及退款退货政策。
- product_consult：商品价格、库存、规格、材质、适用对象、优惠、保质期等售前咨询。
- refund_return：退货、退款、换货、运费谁承担、退款到账、退货条件等退款退货政策。
- after_sale：商品破损、少件、漏发、质量问题、保修、发票、赠品补发等事实售后；不含退款退货政策。
- complaint：投诉、维权、态度差、欺骗、要说法、要求领导处理；只表达不满、没有明确要求建单或转人工。
- human：明确要求建工单、转人工、找客服专员/真人跟进（无论是否带具体问题，只要「要人工/要建单」诉求明确就归这里）。
- chitchat：问候、感谢、闲聊、问你是谁、天气、笑话、告别。
- other：超出电商客服范围、多意图冲突无法判断、没有明确业务动作、拿不准。

边界 few-shot：
- 「拆封了还能退吗」→ refund_return（问政策）
- 「收到的猫粮少了一袋」→ after_sale（陈述事实售后）
- 「订单 1001 到哪了」→ logistics
- 「我要投诉物流」→ complaint（只表达投诉，没要求建单/人工）
- 「帮我建个工单，反馈订单 1001 发货太慢」→ human（明确要建单）
- 「转人工」「找人工客服」→ human（明确要人工）
- 「怎么建工单」→ other（询问流程，不是建单请求）
- 「你们总部在哪」→ other
- 「那个东西怎么回事」→ other（没有明确业务动作）
- 「帮我写首诗」→ other

confidence 口径：明确 0.9+；边界 0.6~0.8；拿不准 <0.6 并优先 other。
禁止把超出范围或拿不准的问题硬塞进业务意图。"""


def intent_messages(question: str, history=None, summary: str = "") -> list:
    history_text = _history_text(history, summary) if history or summary else ""
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
5. 无法判断指代指什么 → 原样返回，不猜。

物流 vs 退款到账（最高优先级，必须遵守）：
- 只看【当前用户消息】的原话用词，不要被上一轮话题带走。
- 原话没有出现「账、退款、退货、退款单、钱、款」时，「什么时候到 / 多久到 / 什么时候能到 / 送到 / 到货」一律指商品包裹送达，补全成「订单X的包裹/物流大概什么时候能到」。
- 即使上一轮客服刚回答过退款到账时间，也按上一条处理。
- 原话出现「到账」，或「退款/退货/钱/款 + 到」时，才指退款到账。
示例：
【最近对话】客服：退款到账时间以支付渠道实际到账为准。
【当前用户消息】那大概什么时候能到
→ resolved_query：订单1001的包裹大概什么时候能到
【当前用户消息】退款多久到账
→ resolved_query：订单1001的退款多久到账"""


def _history_text(history, summary: str = "") -> str:
    lines = []
    if summary:
        lines.append(f"【历史梗概】{summary}")
    for message in history or []:
        role = "用户" if isinstance(message, HumanMessage) else "客服"
        text = message.content if isinstance(message.content, str) else str(message.content or "")
        if text.strip():
            lines.append(f"{role}：{text.strip()}")
    return "\n".join(lines)


def coref_messages(history, current: str, summary: str = "") -> list:
    history_text = _history_text(history, summary) or "（无历史）"
    return [
        SystemMessage(content=COREF_SYSTEM),
        HumanMessage(
            content=f"【最近对话】\n{history_text}\n\n【当前用户消息】\n{current}"
        ),
    ]


class ExpansionOutput(BaseModel):
    # 不用 min_length/max_length 卡 schema：模型偶尔只给 2 条时，让节点补齐到 3 条，
    # 而不是让 Pydantic 直接抛错、整轮退化成单查询。
    queries: list[str] = Field(
        description="严格 3 条侧重点不同的检索查询，不能重复、不能为空"
    )


EXPANSION_SYSTEM = """你是电商售后政策检索的查询扩写器。只输出 queries 一个字段，必须恰好 3 条。

要求：
1. 第 1 条必须是能直接回答「能不能退/适不适用」的核心政策条件查询（精排会用它）；
   三部分别覆盖：核心政策条件 / 时限与起算点 / 例外、运费或办理流程。
2. 保留订单事实（商品名、状态、签收时间），让检索能命中适用条款。
3. 只产出检索词，不回答问题，不编造政策结论。
4. 三条必须互不重复、不能为空字符串。"""


def expansion_messages(question: str, intent: str, order_data: dict | None) -> list:
    import json

    order_text = json.dumps(order_data or {}, ensure_ascii=False, default=str)
    return [
        SystemMessage(content=EXPANSION_SYSTEM),
        HumanMessage(
            content=(
                f"【用户问题】\n{question}\n\n"
                f"【意图】\n{intent}\n\n"
                f"【订单】\n{order_text}"
            )
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

NO_ORDERS_REPLY = "暂无可查询的订单，请转人工协助处理。"
ORDER_NOT_FOUND_REPLY = "未找到该订单，请核对订单号后重试。"

AGENT_SYSTEM_ADDENDUM_KNOWLEDGE = """【本轮是知识类问题】
- 下面已经给出本轮强制检索到的证据；不要再调用 query_faq。
- 只依据证据回答，引用编号沿用 [n] 规则；证据没有的数字、条件、时限不要补。
- 如果还需要订单/物流等业务数据，可以调用 query_order / logistics_query_logistics / query_product。
- 仅当用户明确要求建单时调用 create_ticket；它只会生成一个前端按钮提议，不会立刻建单。仅描述问题、投诉或询问怎么建单不是建单请求。"""

AGENT_SYSTEM_ADDENDUM_BUSINESS = """【本轮是业务数据类问题】
- 订单、物流、商品信息必须调用对应工具查询后再回答，不要凭记忆编造。
- 查询物流前，如果只有订单号，先调用 query_order 拿到物流单号，再调用 logistics_query_logistics。
- 仅当用户明确要求建单时调用 create_ticket；它只会生成一个前端按钮提议，不会立刻建单。
- 普通咨询不要建议转人工。"""

AGENT_SYSTEM_ADDENDUM_HUMAN = """【本轮用户明确要求人工/建单】
- 用户明确要求创建工单时，调用 create_ticket；它只会生成前端确认卡片，不会立刻建单。
- 用户只要求转人工、没有明确要求建单时，不要调用 create_ticket，也不要声称已转接成功；
  说明可以为其创建工单转交人工，并请用户明确说「帮我建个工单」后再操作。
- 用户明确要求建单但还没给出问题描述时，先请用户补充描述，不要编造。"""

AGENT_SYSTEM_ADDENDUM_LOGISTICS = """【本轮是物流查询】
- 用户问的是包裹什么时候送到，不是退款到账时间；不要答退款到账。
- 本轮必须重新调用 logistics_query_logistics 取最新节点（历史里查过也要再查）；只有订单号时先 query_order 拿物流单号。
- 只依据本轮工具返回的节点回答；不要复述历史里的物流结果当最新结果，也不要承诺具体送达时间。
- 仅当用户明确要求建单时调用 create_ticket；普通咨询不要建议转人工。"""


AGENT_SYSTEM_ADDENDUM_REFUND = """【本轮是退款判定】
- 订单数据和退款政策证据已经备齐；不要再调用 query_order / query_faq。
- 只判断「这一单能不能退」，依据必须是订单字段与政策证据。
- 能退 → 调用 submit_refund(order_no)；它只会生成前端退款单按钮，不会立刻建单。
- 不能退 → 说明依据，不要调用 submit_refund。
- 证据不足 → 说明无法确认并建议人工，不要猜。
- 不要追问退款原因；表单会从固定类目里收集。
- 历史里生成过的表单不会留到本轮（前端只在生成它的那条回复下显示按钮）；
  用户本轮再次要求退款时，必须重新调用 submit_refund。
- 只要判断能退，就必须调用 submit_refund；不要在没调用工具时声称「表单已生成/请提交表单」。
- 退款表单只对「普通商品 + 已签收 + 签收后 7 天内」生成；定制类、生鲜类、未签收、已取消的订单不要调用 submit_refund。"""

AGENT_SYSTEM_ADDENDUM_AFTER_SALE = """【本轮是售后判定】
- 订单数据和售后政策证据已经备齐；不要再调用 query_order / query_faq。
- 只判断「这一单能不能走对应售后处理」（发票/保修/补发/破损等），依据是订单字段与政策证据。
- 不要调用 submit_refund。
- 用户明确要求建单时才调用 create_ticket，ticket_type 必须是「售后」；仅描述问题或投诉本身不是建单请求。
- 用户明确要求建单但还没给出问题描述时，先请用户补充描述，不要凭空编造；不能处理 → 说明依据并建议人工。"""

AGENT_SYSTEM_ADDENDUM_CLARIFY = """【本轮意图不明确，需要澄清】
- 先问一个最关键的澄清问题，不要猜用户意图，不要硬塞退款/物流等业务结论。
- 用户补充后按实际诉求回答；能用 query_faq 查政策就用，能查订单就查订单。
- 仍无法判断 → 说明范围并建议转人工，不编造。"""
