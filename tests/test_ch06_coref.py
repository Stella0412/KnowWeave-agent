from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes
from app.prompts.orchestrator import COREF_SYSTEM, CorefOutput


class FakeCorefModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.seen = []

    async def ainvoke(self, messages):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return self.result


def _runtime(model):
    return SimpleNamespace(context=SimpleNamespace(coref_model=model))


def _state(*texts):
    return {"messages": [HumanMessage(content=t) for t in texts]}


@pytest.mark.asyncio
async def test_coref_resolves_reference_with_history():
    model = FakeCorefModel(CorefOutput(resolved_query="我买的猫爬架能退吗"))
    update = await nodes.coref(_state("我买的猫爬架", "它能退吗"), _runtime(model))
    assert update == {
        "resolved_query": "我买的猫爬架能退吗",
        "coref_source": "model",
    }
    assert "我买的猫爬架" in model.seen[0][-1].content
    assert "它能退吗" in model.seen[0][-1].content


@pytest.mark.asyncio
async def test_coref_passthrough_for_complete_question():
    model = FakeCorefModel(CorefOutput(resolved_query="退货政策是什么"))
    update = await nodes.coref(_state("退货政策是什么"), _runtime(model))
    assert update["resolved_query"] == "退货政策是什么"
    assert update["coref_source"] == "passthrough"


@pytest.mark.asyncio
async def test_coref_falls_back_to_raw_on_error():
    model = FakeCorefModel(error=RuntimeError("llm down"))
    update = await nodes.coref(_state("它能退吗"), _runtime(model))
    assert update == {"resolved_query": "它能退吗", "coref_source": "fallback"}


def test_coref_prompt_separates_delivery_from_refund_arrival():
    # 「那大概什么时候能到」紧跟退款答复时，模型容易把「到」消解成退款到账。
    # 产品口径（正确示例）是：不带「账/退款/退货/钱」的「到」一律指包裹送达。
    assert "包裹" in COREF_SYSTEM
    assert "不要被上一轮话题带走" in COREF_SYSTEM
    assert "退款多久到账" in COREF_SYSTEM
    assert "最高优先级" in COREF_SYSTEM


@pytest.mark.asyncio
async def test_coref_without_model_is_passthrough():
    update = await nodes.coref(_state("它能退吗"), _runtime(None))
    assert update["coref_source"] == "passthrough"
