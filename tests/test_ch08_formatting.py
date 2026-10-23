"""ch08 · 中文结果格式化与审计摘要测试。"""

import json

from app.tools.contracts import ToolOutcome


def test_success_payload_shape():
    from app.tools.formatting import format_outcome

    content = format_outcome(ToolOutcome("success", data={"ticket_no": "T-1"}, message=None))
    parsed = json.loads(content)
    assert set(parsed) == {"ok", "category", "data", "message"}
    assert parsed["ok"] is True
    assert parsed["category"] == "success"
    assert parsed["data"] == {"ticket_no": "T-1"}
    assert parsed["message"] is None


def test_not_found_payload_shape():
    from app.tools.formatting import format_outcome

    content = format_outcome(ToolOutcome("not_found", data=None, message="未查询到"))
    parsed = json.loads(content)
    assert parsed["ok"] is True
    assert parsed["category"] == "not_found"
    assert parsed["message"] == "未查询到"


def test_fault_payload_shape():
    from app.tools.formatting import format_outcome

    content = format_outcome(ToolOutcome("fault", data=None, message="boom"))
    parsed = json.loads(content)
    assert parsed["ok"] is False
    assert parsed["category"] == "fault"
    assert parsed["message"] == "boom"


def test_actions_included_only_when_non_empty():
    from app.tools.formatting import format_outcome

    content = format_outcome(ToolOutcome("success", data={"x": 1}, actions=()))
    assert "actions" not in json.loads(content)

    content2 = format_outcome(ToolOutcome("success", data={"x": 1}, actions=({"type": "ticket_preview"},)))
    assert "actions" in json.loads(content2)


def test_truncation_keeps_identity_key_and_stays_valid():
    from app.tools.formatting import format_outcome

    data = {"ticket_no": "T-123", "description": "很长的描述" * 500}
    content = format_outcome(ToolOutcome("success", data=data, message="已创建"), max_chars=200)

    parsed = json.loads(content)
    assert parsed["truncated"] is True
    assert parsed["data"]["ticket_no"] == "T-123"
    assert "description" not in parsed["data"]
    assert len(content) <= 200
    assert parsed["ok"] is True
    assert parsed["category"] == "success"
    assert parsed["message"] == "已创建"


def test_truncation_preserves_multiple_identity_keys():
    from app.tools.formatting import format_outcome

    data = {"tracking_no": "YTO-1", "state": "在途", "return_no": "R-1", "status": "ok",
            "other": "x" * 500}
    content = format_outcome(ToolOutcome("success", data=data), max_chars=150)

    parsed = json.loads(content)
    assert parsed["truncated"] is True
    assert parsed["data"] == {"tracking_no": "YTO-1", "state": "在途", "return_no": "R-1",
                              "status": "ok"}


def test_truncation_with_non_dict_data():
    from app.tools.formatting import format_outcome

    content = format_outcome(ToolOutcome("success", data=["x"] * 1000), max_chars=100)

    parsed = json.loads(content)
    assert parsed["truncated"] is True
    assert parsed["data"] is None


def test_audit_summary_bounded_and_valid():
    from app.tools.formatting import format_outcome, summarize_for_audit

    content = format_outcome(ToolOutcome("success", data={"big": "x" * 5000}), max_chars=100000)
    summary = summarize_for_audit(content, limit=1000)

    parsed = json.loads(summary)
    assert parsed["truncated"] is True
    assert "big" not in parsed
    assert len(summary) <= 1000
    assert parsed["ok"] is True
    assert parsed["category"] == "success"
    assert parsed["message"] is None


def test_audit_summary_within_limit_returns_content():
    from app.tools.formatting import summarize_for_audit

    content = '{"ok": true, "category": "success", "data": {}, "message": null}'
    assert summarize_for_audit(content, limit=1000) == content


def test_non_json_serializable_data_uses_default_str():
    from app.tools.formatting import format_outcome

    content = format_outcome(ToolOutcome("success", data={"when": object()}))
    parsed = json.loads(content)
    assert isinstance(parsed["data"]["when"], str)
