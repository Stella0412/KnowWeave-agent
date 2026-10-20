"""台账服务：本轮判出、人工处置、复发回退、两个率的口径。"""

import pytest

from app.db.repository import FaithCaseRepository
from app.services import rag_eval_service as service


def _latest(run_id="run-1"):
    return {
        "run_id": run_id,
        "samples": 300,
        "models": {"judge": "deepseek-chat"},
        "answers": {"answered": 300},
        "questions": [
            {
                "id": "A1", "bucket": "A_policy", "query": "退款几天到账",
                "answer": "一般 1-5 个工作日到账", "faithful": False,
                "judge_reason": "1-5 个工作日整份资料里都没有",
                "citations": [{"n": 1, "section_path": "退货退款政策 > 退款时效"}],
                "expected_behavior": "no_commitment", "refused": False,
            },
            {
                "id": "D1", "bucket": "D_absent", "query": "有没有宠物保险",
                "answer": "有的，最高赔付 5000 元", "faithful": None,
                "judge_reason": None, "citations": [],
                "expected_behavior": "refuse", "refused": False,
            },
            {
                "id": "A2", "bucket": "A_policy", "query": "怎么申请退款",
                "answer": "在我的订单里申请 [1]", "faithful": True,
                "judge_reason": "有据", "citations": [],
                "expected_behavior": "answer", "refused": False,
            },
        ],
    }


@pytest.fixture
def ledger_env(monkeypatch, session_factory, tmp_path):
    monkeypatch.setattr(service, "SessionFactory", session_factory)
    monkeypatch.setattr(service, "FAITH_SYNC_PATH", tmp_path / "faith_synced.json")
    monkeypatch.setattr(service, "read_latest", lambda: _latest())
    return session_factory


async def _rows(session_factory):
    async with session_factory() as session:
        return await FaithCaseRepository(session).list_cases()


async def test_sync_writes_judge_and_missed_refusal_cases(ledger_env):
    count = await service.sync_faith_cases(force=True)
    assert count == 2
    rows = await _rows(ledger_env)
    assert {row.eval_id for row in rows} == {"A1", "D1"}
    assert all(row.status.value == "未解决" for row in rows)
    assert all(row.seen_count == 1 for row in rows)


async def test_sync_same_run_id_is_idempotent(ledger_env):
    await service.sync_faith_cases(force=True)
    await service.sync_faith_cases()
    rows = await _rows(ledger_env)
    assert all(row.seen_count == 1 for row in rows)


async def test_new_run_reopens_resolved_case_and_bumps_seen_count(ledger_env, monkeypatch):
    await service.sync_faith_cases(force=True)
    await service.resolve_faith_case("A1", "已解决", "补了到账时效条目")

    monkeypatch.setattr(service, "read_latest", lambda: _latest("run-2"))
    await service.sync_faith_cases()
    rows = {row.eval_id: row for row in await _rows(ledger_env)}
    assert rows["A1"].status.value == "未解决"
    assert rows["A1"].seen_count == 2
    assert rows["A1"].resolution is None


async def test_round_metrics_use_explicit_denominator(ledger_env):
    await service.sync_faith_cases(force=True)
    data = await service.list_faith_cases()
    assert data["samples"] == 300
    assert data["round"] == {
        "judged": 1,
        "missed_refusals": 1,
        "resolved": 0,
        "not_needed": 0,
        "open": 2,
        "judged_rate": 2 / 300,
        "confirmed_rate": 0.0,
    }
    assert data["ledger"]["total"] == 2
    by_id = {case["eval_id"]: case for case in data["cases"]}
    assert by_id["D1"]["source"] == "missed_refusal"
    assert by_id["A1"]["source"] == "judge"


async def test_resolve_requires_a_resolution_note(ledger_env):
    await service.sync_faith_cases(force=True)
    with pytest.raises(ValueError, match="必须留一句交代"):
        await service.resolve_faith_case("A1", "已解决", "   ")
    with pytest.raises(ValueError, match="必须留一句交代"):
        await service.resolve_faith_case("A1", "无需解决", None)


async def test_resolve_moves_confirmed_rate(ledger_env):
    await service.sync_faith_cases(force=True)
    await service.resolve_faith_case("A1", "已解决", "补了到账时效条目")
    data = await service.list_faith_cases()
    assert data["round"]["resolved"] == 1
    assert data["round"]["confirmed_rate"] == 1 / 300
    case = next(item for item in data["cases"] if item["eval_id"] == "A1")
    assert case["status"] == "已解决"
    assert case["resolution"] == "补了到账时效条目"


async def test_not_needed_counts_as_judge_false_positive_not_confirmed(ledger_env):
    await service.sync_faith_cases(force=True)
    await service.resolve_faith_case("A1", "无需解决", "1-5 个工作日是提醒，不是承诺")
    data = await service.list_faith_cases()
    assert data["round"]["not_needed"] == 1
    assert data["round"]["resolved"] == 0
    assert data["round"]["confirmed_rate"] == 0.0


async def test_reopen_clears_resolution(ledger_env):
    await service.sync_faith_cases(force=True)
    await service.resolve_faith_case("A1", "已解决", "补了到账时效条目")
    await service.resolve_faith_case("A1", "未解决")
    data = await service.list_faith_cases()
    case = next(item for item in data["cases"] if item["eval_id"] == "A1")
    assert case["status"] == "未解决"
    assert case["resolution"] is None
