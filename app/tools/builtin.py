"""ch08 · 内置工具注册与动态工具快照构建。

内置工具统一注册为 ToolDefinition：invoke 接收 (arguments, ExecutionContext) 返回
ToolOutcome；guard 是可选的本地可信约束（路由白名单、退款资格），在引擎里于
权限放行之后、确认/执行之前应用。

query_logistics 已移交 MCP Server（公开名 logistics_query_logistics），不再注册为
内置工具；submit_refund 的「已选订单一致 + 退款资格」约束整体迁入 guard，invoke
只负责产出前端退款表单动作。
"""

import logging

from app.services.refund_policy import check_refund_eligibility
from app.tools.business import REFUND_REASON_CATEGORIES
from app.tools.contracts import ToolDefinition, ToolOutcome
from app.tools.registry import allowed_tool_names

logger = logging.getLogger(__name__)


def _route_guard(name: str):
    """通用路由守卫：以 ROUTE_TOOL_WHITELIST 为唯一来源，拒绝不在本轮白名单内的内置工具。"""

    def guard(arguments, context):
        state = getattr(context, "state", None) or {}
        allowed = allowed_tool_names(state.get("route"), state.get("intent"))
        if name not in allowed:
            return ToolOutcome("permission_denied", data=None,
                               message=f"本轮不允许调用 {name}")
        return None

    return guard


def _refund_form_action(order_data: dict, order_no: str) -> dict:
    return {
        "type": "refund_form",
        "label": "提交退款单",
        "draft": {
            "order_no": order_no,
            "product_name": (order_data or {}).get("product_name", ""),
            "reason_options": list(REFUND_REASON_CATEGORIES),
        },
    }


def _submit_refund_guard(arguments, context):
    """submit_refund 专属守卫：订单必须与本轮已选订单一致，且满足无理由退货资格。

    返回 ToolOutcome 即拦截（引擎记 权限拒绝）；返回 None 即放行。
    """
    state = getattr(context, "state", None) or {}
    order_no = str(arguments.get("order_no") or "").strip()
    if order_no != str(state.get("order_no") or ""):
        return ToolOutcome("permission_denied", data=None,
                           message="退款单订单号与本轮订单不一致，不能生成表单。")
    eligible, reason = check_refund_eligibility(state.get("order_data") or {})
    if not eligible:
        return ToolOutcome(
            "permission_denied", data=None,
            message=f"该订单不能生成退款表单：{reason}。不要调用 submit_refund，请向用户说明原因。",
        )
    return None


async def _query_order(arguments, context):
    from app.db.repository import OrderRepository

    order_id = arguments.get("order_id")
    async with context.session_factory() as session:
        order = await OrderRepository(session).get(order_id)
    if order is None:
        return ToolOutcome("not_found", data=None,
                           message=f"未找到订单 {order_id}，请核对订单号。")
    signed = order.signed_at.strftime("%Y-%m-%d") if order.signed_at else "未签收"
    return ToolOutcome("success", data=None, message=(
        f"订单 {order.order_no}：状态={order.status}，商品={order.product_name}，"
        f"类型={order.product_type}，金额=¥{order.amount}，"
        f"物流单号={order.tracking_no or '无'}，签收时间={signed}"
    ))


async def _query_product(arguments, context):
    import random

    product_name = arguments.get("product_name")
    return ToolOutcome("success", data=None, message=(
        f"商品「{product_name}」：价格=¥{random.randint(9, 999)}，"
        f"库存={random.randint(0, 500)}件，规格=标准款"
    ))


async def _query_faq(arguments, context):
    from app.tools.business import build_query_faq

    tool = build_query_faq(context.session_factory, context.turn)
    text = await tool.ainvoke(arguments)
    return ToolOutcome("success", data=None, message=text)


async def _submit_refund(arguments, context):
    state = getattr(context, "state", None) or {}
    order_no = str(arguments.get("order_no") or "").strip()
    action = _refund_form_action(state.get("order_data") or {}, order_no)
    return ToolOutcome(
        "success", data=None,
        message=(
            "本轮退款表单已呈现给用户，请简要说明并停止，本轮不要再次调用 submit_refund。"
            "用户下一轮再次要求退款时，表单不会保留，必须重新调用 submit_refund。"
        ),
        actions=(action,),
    )


async def _create_ticket_stub(arguments, context):
    # create_ticket 在 agent 节点被拦截成前端按钮提议（Task 9 前不写库）。
    # 此处仅作为内置注册占位，正常路径不会经引擎调用。
    return ToolOutcome(
        "success", data=None,
        message="工单选项已呈现给用户，请简要说明并停止，不要再次调用 create_ticket。",
    )


def register_builtins(registry) -> None:
    """把原业务工具注册为内置 ToolDefinition。

    每个内置工具都挂通用路由守卫；submit_refund 额外挂退款资格守卫。query_logistics
    不再注册（MCP Server 拥有物流）。
    """
    registry.register(ToolDefinition(
        name="query_order", description="查询订单信息。当用户询问订单状态、订单内容、签收时间时调用。",
        parameters={"type": "object", "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]},
        source="builtin", server=None, original_name=None, invoke=_query_order,
        effect="read", guard=_route_guard("query_order"),
    ))
    registry.register(ToolDefinition(
        name="query_product", description="查询商品信息。当用户询问商品价格、多少钱、库存、规格时调用。",
        parameters={"type": "object", "properties": {"product_name": {"type": "string"}},
                    "required": ["product_name"]},
        source="builtin", server=None, original_name=None, invoke=_query_product,
        effect="read", guard=_route_guard("query_product"),
    ))
    registry.register(ToolDefinition(
        name="query_faq", description="查询常见问题。当用户询问退货/换货/发货/邮费/运费等政策类问题时调用。",
        parameters={"type": "object", "properties": {"keyword": {"type": "string"}},
                    "required": ["keyword"]},
        source="builtin", server=None, original_name=None, invoke=_query_faq,
        effect="read", guard=_route_guard("query_faq"),
    ))
    registry.register(ToolDefinition(
        name="submit_refund", description="提交退款单。仅当判断这一单符合退款条件时调用；前端会展示退款表单。",
        parameters={"type": "object", "properties": {"order_no": {"type": "string"}},
                    "required": ["order_no"]},
        source="builtin", server=None, original_name=None, invoke=_submit_refund,
        effect="read", guard=_submit_refund_guard,
    ))
    registry.register(ToolDefinition(
        name="create_ticket", description="创建人工工单。当用户投诉、维权、或要求转人工等需要人工介入时调用。",
        parameters={"type": "object", "properties": {
            "description": {"type": "string"},
            "ticket_type": {"type": "string", "enum": ["售后", "投诉", "咨询"]},
        }, "required": ["description", "ticket_type"]},
        source="builtin", server=None, original_name=None, invoke=_create_ticket_stub,
        effect="read", guard=_route_guard("create_ticket"),
    ))


async def build_tool_snapshot(registry, provider) -> dict[str, ToolDefinition]:
    """构建本轮工具快照：内置工具 + 已授权 MCP 工具。

    总刷新失败降级为仅内置工具（记 warning），绝不拖垮本轮；公开名与内置重名时
    内置优先，跳过 MCP 重复定义。
    """
    tools = dict(registry.snapshot())
    if provider is None:
        return tools
    try:
        discovered = await provider.refresh()
    except Exception:
        logger.warning("MCP 工具刷新失败，降级为仅内置工具", exc_info=True)
        return tools
    for name, definition in discovered.items():
        if name in tools:
            logger.warning("MCP 工具 %s 与内置工具重名，跳过 MCP 定义", name)
            continue
        tools[name] = definition
    return tools
