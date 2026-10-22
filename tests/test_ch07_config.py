from pathlib import Path

from app.config import Settings
from app.db.models import Conversation, ConversationSummary


def test_ch07_config_defaults():
    s = Settings(_env_file=None)
    assert s.model_context_window == 64000
    assert s.max_output_tokens == 2000
    assert s.max_user_input_tokens == 2000
    assert s.max_agent_steps == 6
    assert s.tool_result_max_tokens == 1200
    assert s.system_prefix_reserve_tokens == 1500
    assert s.evidence_tokens_per_chunk == 200
    assert s.summary_inject_max_tokens == 1000
    assert s.context_safety_margin_tokens == 1250
    assert s.context_target_rounds == 20
    assert s.layer1_share == 0.7
    assert s.steady_turn_tokens == 1200
    assert s.classifier_history_max_tokens == 1200
    assert s.layer2_assistant_prefix_chars == 60
    assert s.order_json_max_tokens == 600
    assert s.cjk_tokens_per_char == 0.6
    assert s.token_estimate_factor == 1.15
    assert s.summary_input_max_tokens == 6000
    assert s.summary_message_max_tokens == 800
    assert s.summary_tool_result_max_tokens == 400
    assert s.summary_max_chars == 200
    assert s.summary_max_output_tokens == 400
    assert s.summary_retry_cooldown_seconds == 60
    assert s.log_file == "log/app.log"


def test_rerank_top_k_alias_wins(monkeypatch):
    monkeypatch.setenv("RERANK_TOP_K", "5")
    monkeypatch.setenv("RERANK_TOP_N", "9")
    assert Settings(_env_file=None).rerank_top_n == 5


def test_ch07_orm_columns_exist():
    assert "summary" in Conversation.__table__.columns
    assert "summary_upto_msg_id" in Conversation.__table__.columns
    assert "layer1_from_msg_id" in Conversation.__table__.columns
    columns = ConversationSummary.__table__.columns
    for name in ("id", "conversation_id", "seq", "from_msg_id", "upto_msg_id", "content", "created_at"):
        assert name in columns


def test_two_step_ddl_files_exist():
    first = Path("sql/ch07-context.sql").read_text(encoding="utf-8")
    second = Path("sql/ch07-layers.sql").read_text(encoding="utf-8")
    assert "summary_upto_msg_id" in first
    assert "layer1_from_msg_id" in second
    assert "conversation_summaries" in second
    assert "UNIQUE KEY" in second
