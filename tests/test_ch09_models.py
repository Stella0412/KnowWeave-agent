from sqlalchemy import text

from app.db.models import (
    AnswerRating,
    LowConfidenceQuestion,
    LowConfidenceSource,
    LowConfidenceStatus,
    ToolAuditLog,
    TurnMetrics,
    TurnStatus,
)


def test_ch09_table_and_column_contract():
    assert {"answer_feedback", "turn_metrics"} <= set(
        LowConfidenceQuestion.metadata.tables
    )
    assert {"message_id", "fingerprint", "merged_into_id", "duplicate_count",
            "status", "citations", "knowledge_chunk_id", "resolution",
            "last_seen_at"} <= {c.name for c in LowConfidenceQuestion.__table__.columns}
    assert "turn_token" in ToolAuditLog.__table__.columns
    assert {"turn_token", "trace_id", "route", "intent", "status", "error_type",
            "llm_calls", "prompt_tokens", "completion_tokens", "total_tokens",
            "usage_estimated", "model_usage", "stages", "cost_cny", "latency_ms",
            "first_token_ms", "refusal_source", "citations_count"} <= {
                c.name for c in TurnMetrics.__table__.columns
            }


def test_ch09_enum_values_are_literals():
    assert {m.value for m in LowConfidenceStatus} == {"待处理", "已归并", "已关闭", "已忽略"}
    assert {m.value for m in AnswerRating} == {"up", "down"}
    assert {m.value for m in TurnStatus} == {"ok", "error", "interrupted", "cancelled"}


async def test_ch09_low_conf_defaults_and_roundtrip(session_factory):
    async with session_factory() as session:
        row = LowConfidenceQuestion(
            raw_question="保价规则是什么", source=LowConfidenceSource.SELF_CHECK,
        )
        session.add(row)
        await session.commit()
        assert row.fingerprint == ""
        assert row.duplicate_count == 1
        assert row.status is LowConfidenceStatus.PENDING
        raw = await session.scalar(
            text("select status from low_confidence_questions where id=:i"), {"i": row.id}
        )
    assert raw == "待处理"
