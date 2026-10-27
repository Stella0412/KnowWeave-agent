"""意图成本账：按 intent 标签汇总 token/成本，产物落盘，页面只读产物。"""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.observability import cost_report as cost
from app.routers import observability as obs_router


def _tag_row(tags, tokens, total_cost):
    return {"tags": tags, "sum_totalTokens": tokens, "sum_totalCost": total_cost}


def _root_row(tags, count):
    return {"tags": tags, "count_count": count}


def test_aggregate_splits_intents_and_untagged():
    report = cost.aggregate(
        [
            _tag_row(["chat", "mode:agent", "intent:logistics"], 100, 0.001),
            _tag_row(["chat", "mode:stream", "intent:product_consult"], 300, 0.004),
            _tag_row(["chat", "mode:agent"], 50, 0.0005),
        ],
        [
            _root_row(["chat", "mode:agent", "intent:logistics"], 2),
            _root_row(["chat", "mode:stream", "intent:product_consult"], 1),
            _root_row(["chat", "mode:agent"], 2),
        ],
        window_days=7,
        generated_at="2026-10-27T18:17:00",
        environment="development",
    )

    assert [e["intent"] for e in report["intents"]] == [
        "product_consult", "logistics",
    ]
    top, second = report["intents"]
    assert top["label"] == "商品咨询"
    assert top["requests"] == 1
    assert top["tokens"] == 300
    assert top["avg_tokens"] == 300
    assert top["share"] == 0.75
    assert top["priced"] is True
    assert second["avg_tokens"] == 50
    assert second["share"] == 0.25

    # 总量只算意图确定之后；分类前的开销单独列，不混进任何意图
    assert report["totals"] == {
        "requests": 3, "tokens": 400, "cost": 0.005, "avg_tokens": 133,
    }
    assert report["untagged"] == {"requests": 2, "tokens": 50, "cost": 0.0005}
    assert report["top_intent"]["intent"] == "product_consult"
    assert report["top_intent"]["ranked_by"] == "cost"
    assert report["warnings"] == []


def test_aggregate_warns_when_tokens_have_no_price():
    report = cost.aggregate(
        [_tag_row(["chat", "intent:refund_return"], 120, 0)],
        [_root_row(["chat", "intent:refund_return"], 1)],
        window_days=7,
        generated_at="2026-10-27T18:17:00",
        environment="development",
    )
    entry = report["intents"][0]
    assert entry["priced"] is False
    # 全都没价格时，最烧钱意图退化为按 token 排
    assert report["top_intent"]["ranked_by"] == "tokens"
    assert any("退款退货" in warning for warning in report["warnings"])


def test_aggregate_without_intent_rows_explains_why():
    report = cost.aggregate(
        [], [], window_days=7, generated_at="2026-10-27T18:17:00",
        environment="development",
    )
    assert report["intents"] == []
    assert report["totals"]["tokens"] == 0
    assert report["top_intent"] is None
    assert report["warnings"] == ["窗口内没有带意图标签的请求，先聊几轮再看"]


def test_build_report_queries_langfuse_by_environment_and_root_span():
    queries = []

    def fake_query(query):
        queries.append(query)
        if any(f["column"] == "name" for f in query["filters"]):
            return [_root_row(["chat", "intent:order"], 1)]
        return [_tag_row(["chat", "intent:order"], 80, 0.0002)]

    report = cost.build_report(
        7, now=datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc),
        query=fake_query, environment="development",
    )
    assert report["totals"]["tokens"] == 80
    assert report["generated_at"]
    assert len(queries) == 2

    tag_query, root_query = queries
    assert tag_query["fromTimestamp"] == "2026-10-20T10:00:00Z"
    assert tag_query["toTimestamp"] == "2026-10-27T10:00:00Z"
    assert {"type": "string", "column": "environment", "operator": "=",
            "value": "development"} in tag_query["filters"]
    assert {"type": "string", "column": "name", "operator": "=",
            "value": "handle-chat-turn"} in root_query["filters"]
    assert tag_query["dimensions"] == [{"field": "tags"}]


def test_refresh_writes_product_and_reads_it_back(tmp_path):
    path = tmp_path / "cost_report" / "latest.json"
    report = cost.refresh(
        7, path=path,
        query=lambda q: [_tag_row(["chat", "intent:order"], 10, 0)],
        now=datetime(2026, 10, 27, tzinfo=timezone.utc),
    )
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert cost.read_report(path)["window_days"] == 7


def test_read_report_missing_or_broken_returns_none(tmp_path):
    assert cost.read_report(tmp_path / "nope.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert cost.read_report(broken) is None


async def test_cost_report_endpoints_read_product_and_rerun(monkeypatch, tmp_path):
    monkeypatch.setattr(cost, "REPORT_PATH", tmp_path / "latest.json")
    app = FastAPI()
    app.include_router(obs_router.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/observability/api/cost-report")
        bad_days = await client.post(
            "/observability/api/cost-report/rerun", json={"days": 2},
        )

        product = {
            "window_days": 7, "totals": {"tokens": 42}, "intents": [],
            "warnings": [],
        }
        cost.write_report(product, cost.REPORT_PATH)
        loaded = await client.get("/observability/api/cost-report")

        rerun = {**product, "window_days": 1, "totals": {"tokens": 7}}
        monkeypatch.setattr(cost, "refresh", lambda days: rerun)
        refreshed = await client.post(
            "/observability/api/cost-report/rerun", json={"days": 1},
        )

    assert missing.status_code == 404
    assert bad_days.status_code == 400
    assert loaded.status_code == 200
    assert loaded.json()["totals"]["tokens"] == 42
    assert refreshed.status_code == 200
    assert refreshed.json()["window_days"] == 1


async def test_rerun_reports_langfuse_failure_as_503(monkeypatch):
    def boom(days):
        raise cost.CostReportError("连不上 Langfuse：ConnectError")

    monkeypatch.setattr(cost, "refresh", boom)
    app = FastAPI()
    app.include_router(obs_router.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/observability/api/cost-report/rerun", json={"days": 7},
        )
    assert resp.status_code == 503
    assert "连不上 Langfuse" in resp.json()["detail"]
