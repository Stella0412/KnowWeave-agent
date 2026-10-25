import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.db.models import TurnMetrics
from app.db.repository import ChatRepository
from app.services.chat_service import ChatService


class FakeGraph:
    def __init__(self, result):
        self.result = result

    async def aget_state(self, config):
        return SimpleNamespace(values={}, tasks=[])

    async def ainvoke(self, graph_input, config=None, context=None):
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def astream(self, graph_input, config=None, context=None, stream_mode=None):
        if isinstance(self.result, BaseException):
            raise self.result
        yield ("updates", {"fallback_reply": {"answer": "答"}})


class FakeHandler:
    def finish_turn(self, token):
        return {
            "llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15, "usage_estimated": False,
            "model_usage": {"m": {"calls": 1, "prompt_tokens": 10,
                                  "completion_tokens": 5, "cost_cny": 0.0,
                                  "estimated": False, "priced": False}},
            "cost_cny": 0.0, "first_token_ms": 12, "unpriced_models": ["m"],
            "stages": {}, "trace_id": "trace-1",
        }


class FakeWriter:
    def __init__(self):
        self.rows = []

    async def write(self, payload):
        self.rows.append(payload)


async def _service(session_factory, graph):
    writer = FakeWriter()
    return ChatService(
        model=object(), intent_model=object(), session_factory=session_factory,
        graph=graph, metrics_handler=FakeHandler(), turn_metrics_writer=writer,
    ), writer


async def test_run_writes_ok_metrics(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    graph = FakeGraph({
        "messages": [AIMessage(content="答")], "answer": "答",
        "route": "business", "intent": "faq", "citations": [{"n": 1}],
        "suggested_actions": [],
    })
    service, writer = await _service(session_factory, graph)
    result = await service.run(str(cid), "问")
    assert result["message_id"] is not None
    assert writer.rows[0]["status"] == "ok"
    assert writer.rows[0]["first_token_ms"] is None   # 非流式
    assert writer.rows[0]["citations_count"] == 1


async def test_run_writes_error_metrics(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    service, writer = await _service(session_factory, FakeGraph(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        await service.run(str(cid), "问")
    assert writer.rows[0]["status"] == "error"
    assert writer.rows[0]["error_type"] == "RuntimeError"


async def test_stream_cancelled_is_interrupted(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    service, writer = await _service(session_factory, FakeGraph(asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        async for _ in service.stream(str(cid), "问"):
            pass
    assert writer.rows[0]["status"] == "interrupted"
