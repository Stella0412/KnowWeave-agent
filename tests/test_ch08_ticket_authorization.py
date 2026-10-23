"""ch08 · 建单意图证据单元测试。

证据只来自落库用户消息文本；模型输出、消解文本、工具参数、create_ticket 参数
一律不能产生授权。这里覆盖 update_ticket_request 纯函数语义 + ChatService 状态
隔离与 JSON 可序列化。
"""

import json

import pytest

from app.config import Settings
from app.db.repository import ChatRepository
from app.services.chat_service import ChatService
from app.tools.ticket_authorization import (
    TicketRequestEvidence,
    update_ticket_request,
)


# —— 纯函数：update_ticket_request ——


def test_explicit_request_active_with_message_id_and_quote():
    ev = update_ticket_request(None, 42, "帮我建个工单")
    assert ev is not None
    assert ev.status == "active"
    assert ev.message_id == 42
    assert ev.quote == "帮我建个工单"
    assert ev.quote in "帮我建个工单"


def test_missing_description_still_active_evidence():
    # 明确请求但缺少描述：证据仍为 active；描述由后续消息补充，不改变证据归属。
    ev = update_ticket_request(None, 7, "帮我提个工单")
    assert ev is not None
    assert ev.status == "active"
    assert ev.message_id == 7
    assert ev.quote == "帮我提个工单"


def test_supplement_keeps_original_evidence():
    first = update_ticket_request(None, 1, "帮我建个工单")
    second = update_ticket_request(first, 2, "收到的猫粮包装破损")
    assert second is not None
    assert second.status == "active"
    assert second.message_id == 1
    assert second.quote == first.quote


def test_new_explicit_request_refreshes_message_id_and_quote():
    first = update_ticket_request(None, 1, "帮我建个工单")
    refreshed = update_ticket_request(first, 3, "给我建工单")
    assert refreshed.status == "active"
    assert refreshed.message_id == 3
    assert refreshed.quote == "给我建工单"


def test_withdrawal_cancels_and_is_not_reusable():
    first = update_ticket_request(None, 1, "帮我建个工单")
    cancelled = update_ticket_request(first, 2, "算了不建工单")
    assert cancelled.status == "cancelled"
    # 撤回后普通补充/寒暄不能复活证据
    assert update_ticket_request(cancelled, 3, "收到货了") is None
    assert update_ticket_request(cancelled, 4, "知道了") is None


def test_later_explicit_request_after_cancel_starts_new_active():
    first = update_ticket_request(None, 1, "帮我建个工单")
    cancelled = update_ticket_request(first, 2, "不要建了")
    assert cancelled.status == "cancelled"
    revived = update_ticket_request(cancelled, 5, "还是帮我建个工单吧")
    assert revived.status == "active"
    assert revived.message_id == 5


def test_standalone_negation_produces_no_evidence():
    assert update_ticket_request(None, 1, "不要帮我建工单") is None


@pytest.mark.parametrize("marker", ["不用了", "不要了", "算了", "取消", "撤销", "撤回"])
def test_bare_withdrawal_after_active_is_cancelled(marker):
    first = update_ticket_request(None, 1, "帮我建个工单")
    assert first.status == "active"
    cancelled = update_ticket_request(first, 2, marker)
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.message_id == 2


@pytest.mark.parametrize("marker", ["算了", "不用了", "不要了", "取消"])
def test_standalone_bare_withdrawal_without_prior_is_none(marker):
    # 没有 active 前值时，裸撤回只是寒暄/终止语，不产生任何证据。
    assert update_ticket_request(None, 1, marker) is None


def test_new_explicit_request_wins_over_bare_withdrawal():
    first = update_ticket_request(None, 1, "帮我建个工单")
    revived = update_ticket_request(first, 3, "算了，还是帮我建个工单吧")
    assert revived.status == "active"
    assert revived.message_id == 3


@pytest.mark.parametrize(
    "text",
    ["能不能帮我建个工单", "可否帮我建个工单", "是否帮我建个工单", "是不是要帮我建个工单"],
)
def test_extended_question_markers_return_none(text):
    assert update_ticket_request(None, 1, text) is None


def test_consumed_never_reused_even_with_filler():
    consumed = TicketRequestEvidence(message_id=1, quote="帮我建个工单", status="consumed")
    assert update_ticket_request(consumed, 2, "收到货了") is None
    assert update_ticket_request(consumed, 3, "帮我补充一下描述") is None


def test_update_never_produces_consumed():
    first = update_ticket_request(None, 1, "帮我建个工单")
    second = update_ticket_request(first, 2, "收到的猫粮包装破损")
    assert second.status != "consumed"


@pytest.mark.parametrize(
    "text",
    [
        "他说‘帮我建个工单’，这是什么意思？",
        "朋友让我建工单",
        "如果猫粮坏了就帮我建个工单",
        "怎么建工单？",
        "我要投诉",
        "收到的猫粮包装破损",
    ],
)
def test_non_authorizing_texts_return_none(text):
    assert update_ticket_request(None, 1, text) is None


def test_quote_excerpt_is_bounded():
    long_tail = "帮我建个工单，" + "猫粮包装破损，" * 100
    ev = update_ticket_request(None, 1, long_tail)
    assert ev.status == "active"
    assert len(ev.quote) <= 200


# —— 状态层：ChatService 注入与隔离 ——


class _Snapshot:
    def __init__(self, values):
        self.values = values
        self.tasks = []


class _ThreadedFakeGraph:
    """模拟按 conversation 隔离的 checkpointer：每个 thread_id 各自保存最近一次输入。"""

    def __init__(self):
        self._state = {}
        self.inputs = []

    async def aget_state(self, config):
        tid = config["configurable"]["thread_id"]
        return _Snapshot(dict(self._state.get(tid, {})))

    async def ainvoke(self, input, config, context):
        tid = config["configurable"]["thread_id"]
        self._state[tid] = dict(input)
        self.inputs.append((tid, dict(input)))
        return dict(input)


def _service(session_factory, graph, **overrides):
    s = Settings(_env_file=None, **overrides)
    return ChatService(
        model=object(), intent_model=object(), session_factory=session_factory,
        graph=graph, settings=s,
    )


@pytest.mark.asyncio
async def test_stored_state_value_is_json_serializable_dict(session_factory):
    graph = _ThreadedFakeGraph()
    service = _service(session_factory, graph)
    result = await service.run(None, "帮我建个工单")
    cid = result["session_id"]

    ticket = graph.inputs[-1][1].get("ticket_request")
    assert isinstance(ticket, dict)
    json.dumps(ticket)  # 不抛异常即 JSON 可序列化
    assert ticket["status"] == "active"

    async with session_factory() as session:
        rows = await ChatRepository(session).list_messages(cid)
    user_msg = [r for r in rows if r.role.value == "user"][0]
    assert ticket["message_id"] == user_msg.id


@pytest.mark.asyncio
async def test_different_conversations_do_not_share_evidence(session_factory):
    graph = _ThreadedFakeGraph()
    service = _service(session_factory, graph)

    # 会话 1：明确请求 → active
    r1 = await service.run(None, "帮我建个工单")
    ev1 = graph.inputs[-1][1].get("ticket_request")
    assert ev1 is not None and ev1["status"] == "active"

    # 会话 1 第二句：补充描述，证据保持原 message_id/quote
    r2 = await service.run(str(r1["session_id"]), "收到的猫粮包装破损")
    ev1b = graph.inputs[-1][1].get("ticket_request")
    assert ev1b["status"] == "active"
    assert ev1b["message_id"] == ev1["message_id"]
    assert ev1b["quote"] == ev1["quote"]

    # 会话 2：全新会话，仅投诉 → 不继承会话 1 的 active
    await service.run(None, "我要投诉")
    ev2 = graph.inputs[-1][1].get("ticket_request")
    assert ev2 is None
