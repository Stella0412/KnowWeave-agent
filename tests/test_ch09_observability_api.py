from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.db.models import AnswerFeedback, TurnMetrics
from app.routers import observability as obs_router
from app.services.observability_service import ObservabilityService, percentile


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 95) == 4


async def _seed(session_factory):
    now = datetime.now()
    async with session_factory() as session:
        session.add_all([
            TurnMetrics(
                conversation_id=1, turn_token="t1", trace_id="tr1",
                route="business", intent="faq", status="ok", llm_calls=2,
                prompt_tokens=100, completion_tokens=20, total_tokens=120,
                usage_estimated=False,
                model_usage={"m": {"calls": 1, "prompt_tokens": 100,
                                   "completion_tokens": 20, "cost_cny": 0,
                                   "estimated": False, "priced": False}},
                stages={"embedding": {"model": "e", "calls": 1, "duration_ms": 10,
                                      "items": 2, "tokens": 30, "estimated": True}},
                cost_cny=0, latency_ms=1000, first_token_ms=500,
                refusal_source=None, citations_count=2, created_at=now,
            ),
            TurnMetrics(
                conversation_id=2, turn_token="t2", trace_id=None,
                route="business", intent="faq", status="error", llm_calls=1,
                prompt_tokens=50, completion_tokens=10, total_tokens=60,
                usage_estimated=True, model_usage=None, stages=None,
                cost_cny=0, latency_ms=3000, first_token_ms=None,
                refusal_source="self_check", citations_count=0, created_at=now,
            ),
        ])
        session.add_all([
            AnswerFeedback(conversation_id=1, message_id=10, rating="up"),
            AnswerFeedback(conversation_id=2, message_id=11, rating="down"),
        ])
        await session.commit()


async def test_summary_turns_tools(session_factory):
    await _seed(session_factory)
    service = ObservabilityService(session_factory)
    app = FastAPI()
    app.include_router(obs_router.router)
    app.state.observability_service = service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        summary = await client.get("/observability/api/summary?days=7")
        turns = await client.get("/observability/api/turns?days=7&status=ok")
        tools = await client.get("/observability/api/tools?days=7")
        bad = await client.get("/observability/api/summary?days=2")
        bad_status = await client.get("/observability/api/turns?status=nope")
    assert summary.json()["turns"] == 2
    assert summary.json()["satisfaction"]["rate"] == 0.5
    assert summary.json()["error_rate"] == 0.5
    assert summary.json()["latency_ms"]["p50"] in (1000, 3000)
    assert turns.json()["total"] == 1
    assert turns.json()["items"][0]["turn_token"] == "t1"
    assert tools.status_code == 200
    assert bad.status_code == 400
    assert bad_status.status_code == 400


async def test_observability_works_without_langfuse(session_factory):
    # 本测试只验证本地聚合不访问 Langfuse；服务无 client 依赖。
    await _seed(session_factory)
    summary = await ObservabilityService(session_factory).summary(7)
    assert summary["turns"] == 2
