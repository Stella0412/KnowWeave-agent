from app.prompts.context import (
    build_dynamic_block,
    compact_order_json,
    truncate_evidence,
)
from app.services.token_counter import TokenCounter


def _counter():
    return TokenCounter(estimate_factor=1.0)


def test_dynamic_block_has_all_sections_in_fixed_order():
    block = build_dynamic_block(
        route="refund_flow",
        intent="refund_return",
        summary="用户买过猫爬架，订单1001。",
        evidence="[1] 七天无理由",
        order_data={"order_no": "1001", "product_name": "猫爬架"},
        counter=_counter(),
        evidence_budget_tokens=1000,
        order_max_tokens=600,
    )
    assert block.startswith("【本轮上下文资料｜仅本轮有效】")
    order = ["【历史梗概】", "【检索证据】", "【本轮路由】", "【本轮订单】", "【判定优先级】"]
    positions = [block.index(item) for item in order]
    assert positions == sorted(positions)
    assert "七天无理由" in block
    assert "1001" in block


def test_knowledge_route_marks_empty_evidence_explicitly():
    block = build_dynamic_block(
        route="knowledge", intent="product_consult", summary="", evidence="",
        order_data=None, counter=_counter(),
        evidence_budget_tokens=1000, order_max_tokens=600,
    )
    assert "【检索证据】\n（无）" in block


def test_empty_unknown_route_has_no_block():
    block = build_dynamic_block(
        route="", intent=None, summary="", evidence="", order_data=None,
        counter=_counter(), evidence_budget_tokens=1000, order_max_tokens=600,
    )
    assert block == ""


def test_evidence_truncates_by_complete_citation_blocks():
    text = "[1] " + "甲" * 20 + "\n[2] " + "乙" * 20 + "\n[3] " + "丙" * 20
    counter = _counter()
    out = truncate_evidence(text, max_tokens=30, counter=counter)
    assert "[1]" in out
    assert "证据已截断" in out
    assert counter.count_text(out) <= 30


def test_order_json_drops_low_priority_fields_and_marks():
    order = {
        "order_no": "1001", "product_name": "猫爬架", "product_type": "普通",
        "status": "已签收", "signed_at": "2026-10-01T10:00:00",
        "amount": 199, "tracking_no": "SF1001CN", "paid_at": "2026-09-28T10:00:00",
    }
    text = compact_order_json(order, max_tokens=40, counter=_counter())
    assert "1001" in text
    assert "__truncated__" in text
    assert "paid_at" not in text
