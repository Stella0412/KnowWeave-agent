from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.graph.nodes import refund_expand
from app.prompts.orchestrator import ExpansionOutput


class FakeExpansionModel:
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
    return SimpleNamespace(context=SimpleNamespace(expansion_model=model))


def _state(**overrides):
    state = {
        "resolved_query": "订单1001的猫爬架能退吗",
        "intent": "refund_return",
        "order_data": {"order_no": "1001", "product_name": "猫爬架", "status": "已签收"},
        "messages": [HumanMessage(content="这个能退吗")],
        "flow_steps": [],
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_expand_returns_exactly_three_unique_queries():
    model = FakeExpansionModel(ExpansionOutput(queries=["退货条件", "签收后几天", "定制商品例外"]))
    update = await refund_expand(_state(), _runtime(model))
    assert update["policy_queries"] == ["退货条件", "签收后几天", "定制商品例外"]
    assert update["expand_degraded"] is False
    assert "猫爬架" in model.seen[0][-1].content


@pytest.mark.asyncio
async def test_expand_fills_duplicates_to_three_and_marks_degraded():
    model = FakeExpansionModel(ExpansionOutput(queries=["退货条件", "退货条件", "  "]))
    update = await refund_expand(_state(), _runtime(model))
    assert len(update["policy_queries"]) == 3
    assert len(set(update["policy_queries"])) == 3
    assert update["expand_degraded"] is True


@pytest.mark.asyncio
async def test_expand_failure_degrades_to_single_query():
    model = FakeExpansionModel(error=RuntimeError("llm down"))
    update = await refund_expand(_state(), _runtime(model))
    assert update["policy_queries"] == ["订单1001的猫爬架能退吗"]
    assert update["expand_degraded"] is True
