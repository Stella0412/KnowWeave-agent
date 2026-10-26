from langchain_core.callbacks import BaseCallbackHandler
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


def test_no_langfuse_probe_when_disabled(monkeypatch):
    import langfuse

    def boom():
        raise AssertionError("langfuse.get_client must not be called when disabled")

    monkeypatch.setattr(langfuse, "get_client", boom)
    handler = _handler()  # no trace_id_provider -> never touches langfuse
    handler.on_chain_start({}, {}, run_id="chain", metadata={"turn_token": "t"})
    handler.on_chat_model_start(
        {"name": "m", "kwargs": {"model": "m"}}, [[AIMessage(content="问")]],
        run_id="llm", parent_run_id="chain", metadata={"turn_token": "t"},
    )
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="答"))]]),
        run_id="llm",
    )
    assert handler.finish_turn("t")["trace_id"] is None


def test_trace_id_provider_used_when_set():
    handler = TurnMetricsHandler(
        prices=PriceBook({"m": ModelPrice(1.0, 2.0)}),
        counter=TokenCounter(estimate_factor=1.0),
        trace_id_provider=lambda: "trace-123",
    )
    handler.on_chain_start({}, {}, run_id="chain", metadata={"turn_token": "t"})
    assert handler.finish_turn("t")["trace_id"] == "trace-123"


def test_record_stage_shape_and_first_model():
    handler = _handler()
    handler.record_stage("t", "embedding", model="e1", duration_ms=12, texts=["问"], items=1)
    handler.record_stage("t", "embedding", model="e2", duration_ms=8, texts=["问2"], items=2)
    snap = handler.finish_turn("t")
    stage = snap["stages"]["embedding"]
    assert stage["model"] == "e1"
    assert stage["calls"] == 2
    assert stage["duration_ms"] == 20
    assert stage["items"] == 3
    assert stage["tokens"] > 0


def test_llm_output_token_usage_fallback():
    handler = _handler()
    handler.on_chain_start({}, {}, run_id="chain", metadata={"turn_token": "t"})
    handler.on_chat_model_start(
        {"name": "m", "kwargs": {"model": "m"}}, [[AIMessage(content="问")]],
        run_id="llm", parent_run_id="chain", metadata={"turn_token": "t"},
    )
    handler.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="答"))]],
            llm_output={"token_usage": {
                "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
            }},
        ),
        run_id="llm",
    )
    snap = handler.finish_turn("t")
    assert snap["prompt_tokens"] == 7
    assert snap["completion_tokens"] == 3
    assert snap["total_tokens"] == 10
    assert snap["usage_estimated"] is False


def test_handler_is_langchain_callback_compatible():
    """回调管理器直接读取 run_inline/raise_error，裸 handler 会让每轮对话 500。"""
    handler = _handler()
    assert isinstance(handler, BaseCallbackHandler)
    assert handler.run_inline is False
    assert handler.raise_error is False


def test_chat_openai_model_name_is_attributed_and_priced():
    handler = _handler()
    handler.on_chat_model_start(
        {"name": "ChatOpenAI", "kwargs": {"model_name": "m"}},
        [[AIMessage(content="问")]],
        run_id="llm", metadata={"turn_token": "t"},
    )
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(
            message=AIMessage(content="答", usage_metadata={
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            })
        )]]),
        run_id="llm",
    )
    snap = handler.finish_turn("t")
    assert list(snap["model_usage"]) == ["m"]
    assert snap["cost_cny"] == 0.00002
    assert snap["unpriced_models"] == []
