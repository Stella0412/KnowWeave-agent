from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.observability.handler import TurnMetricsHandler
from app.observability.prices import ModelPrice, PriceBook
from app.services.token_counter import TokenCounter


class FakeMessage:
    def __init__(self, usage=None, content="答"):
        self.usage_metadata = usage
        self.content = content


class FakeGeneration:
    def __init__(self, usage=None, content="答"):
        self.message = FakeMessage(usage, content)
        self.text = content


def _handler():
    return TurnMetricsHandler(
        prices=PriceBook({"m": ModelPrice(1.0, 2.0)}),
        counter=TokenCounter(estimate_factor=1.0),
    )


def test_interleaved_turns_are_isolated():
    handler = _handler()
    for token in ("a", "b"):
        handler.on_chain_start({}, {}, run_id=token, parent_run_id=None,
                               metadata={"turn_token": token})
        handler.on_chat_model_start(
            {"name": "m", "kwargs": {"model": "m"}}, [[AIMessage(content="问")]],
            run_id=f"llm-{token}", parent_run_id=token,
            metadata={"turn_token": token},
        )
        handler.on_llm_end(
            LLMResult(generations=[[ChatGeneration(
                message=AIMessage(content="答", usage_metadata={
                    "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                })
            )]]),
            run_id=f"llm-{token}",
        )
    a = handler.finish_turn("a")
    b = handler.finish_turn("b")
    assert a["prompt_tokens"] == b["prompt_tokens"] == 10
    assert a["completion_tokens"] == b["completion_tokens"] == 5
    assert a["cost_cny"] == b["cost_cny"] == 0.00002
    assert handler.finish_turn("a") is None


def test_missing_usage_falls_back_to_estimate():
    handler = _handler()
    handler.on_chain_start({}, {}, run_id="chain", metadata={"turn_token": "t"})
    handler.on_chat_model_start(
        {"name": "m", "kwargs": {"model": "m"}}, [[AIMessage(content="问")]],
        run_id="llm", parent_run_id="chain", metadata={"turn_token": "t"},
    )
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="答"))]]),
        run_id="llm",
    )
    snap = handler.finish_turn("t")
    assert snap["usage_estimated"] is True
    assert snap["prompt_tokens"] > 0
    assert snap["completion_tokens"] > 0


def test_first_token_and_unknown_price():
    handler = _handler()
    handler.on_chain_start({}, {}, run_id="chain", metadata={"turn_token": "t"})
    handler.on_chat_model_start(
        {"name": "x", "kwargs": {"model": "x"}}, [[AIMessage(content="问")]],
        run_id="llm", parent_run_id="chain", metadata={"turn_token": "t"},
    )
    handler.on_llm_new_token("首", run_id="llm")
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="答"))]]),
        run_id="llm",
    )
    snap = handler.finish_turn("t")
    assert snap["first_token_ms"] is not None
    assert snap["unpriced_models"] == ["x"]
    assert snap["cost_cny"] == 0
