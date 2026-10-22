import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings
from app.services.token_counter import (
    ContextBudgetError,
    TokenCounter,
    compute_budget,
)


def test_demo_budget_is_exact():
    s = Settings(
        _env_file=None,
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_n=5,
    )
    b = compute_budget(s)
    assert (b.fixed, b.peak, b.window_history, b.history) == (8750, 3600, 5650, 5650)
    assert (b.layer1, b.layer2) == (3954, 1695)


def test_default_budget_is_exact():
    b = compute_budget(Settings(_env_file=None))
    assert (b.fixed, b.peak, b.window_history, b.history) == (9750, 7200, 47050, 34286)
    assert (b.layer1, b.layer2) == (24000, 10285)


def test_impossible_window_raises():
    s = Settings(_env_file=None, model_context_window=3000)
    with pytest.raises(ContextBudgetError, match="上下文预算不足"):
        compute_budget(s)


def test_invalid_layer1_share_raises():
    s = Settings(_env_file=None, layer1_share=1.0)
    with pytest.raises(ContextBudgetError, match="LAYER1_SHARE"):
        compute_budget(s)


def test_cjk_count_and_factor_calibration():
    base = TokenCounter(cjk_tokens_per_char=1.0, estimate_factor=1.0)
    boosted = TokenCounter(cjk_tokens_per_char=1.0, estimate_factor=1.5)
    assert base.count_text("你好") == 2
    assert boosted.count_text("你好") == 3
    assert base.count_text("") == 0
    assert base.count_text("你好" * 100) > base.count_text("你好")


def test_message_count_includes_protocol_and_tool_call_args():
    counter = TokenCounter(estimate_factor=1.0)
    plain = HumanMessage(content="你好")
    assert counter.count_message(plain) == 4 + 2
    ai = AIMessage(
        content="",
        tool_calls=[{
            "name": "query_order", "args": {"order_id": "1001"},
            "id": "call_1", "type": "tool_call",
        }],
    )
    tool = ToolMessage(content="结果", tool_call_id="call_1")
    assert counter.count_message(ai) > 4
    assert counter.count_message(tool) > 4 + 1


def test_truncate_text_respects_budget_and_marks():
    counter = TokenCounter(estimate_factor=1.0)
    out = counter.truncate_text("甲" * 100, 20, "…（已截断）")
    assert counter.count_text(out) <= 20
    assert out.endswith("…（已截断）")
