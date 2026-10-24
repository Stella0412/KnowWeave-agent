import pytest

from app.tools.contracts import ExecutionContext
from app.tools.ticket_confirmation import TicketDraft, confirm_ticket


def _draft() -> TicketDraft:
    return TicketDraft.create(
        conversation_id=1,
        tool_call_id="call-1",
        ticket_type="售后",
        description="猫粮包装破损",
        request_message_id=10,
    )


def test_confirmation_id_is_deterministic_and_bound_to_content():
    first = _draft()
    second = _draft()
    changed = TicketDraft.create(
        conversation_id=1,
        tool_call_id="call-1",
        ticket_type="售后",
        description="内容被篡改",
        request_message_id=10,
    )

    assert first.confirmation_id == second.confirmation_id
    assert first.confirmation_id != changed.confirmation_id


def test_confirm_ticket_uses_interrupt_result(monkeypatch):
    seen = []

    def fake_interrupt(payload):
        seen.append(payload)
        return {"decision": "confirm", "confirmation_id": payload["confirmation_id"]}

    monkeypatch.setattr("app.tools.ticket_confirmation.interrupt", fake_interrupt)
    context = ExecutionContext(1, "call-1")

    assert confirm_ticket(_draft(), context) is True
    assert seen[0]["type"] == "ticket_confirmation"
    assert seen[0]["draft"]["description"] == "猫粮包装破损"


def test_cancel_and_mismatched_confirmation_are_rejected(monkeypatch):
    draft = _draft()
    context = ExecutionContext(1, "call-1")
    monkeypatch.setattr(
        "app.tools.ticket_confirmation.interrupt",
        lambda payload: {"decision": "cancel", "confirmation_id": payload["confirmation_id"]},
    )
    assert confirm_ticket(draft, context) is False

    monkeypatch.setattr(
        "app.tools.ticket_confirmation.interrupt",
        lambda payload: {"decision": "confirm", "confirmation_id": "tampered"},
    )
    assert confirm_ticket(draft, context) is False
