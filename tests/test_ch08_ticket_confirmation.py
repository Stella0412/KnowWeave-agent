import pytest

from app.tools.contracts import ExecutionContext
from app.tools.ticket_confirmation import TicketDraft, confirm_ticket
from app.tools.ticket_confirmation import draft_from_call


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


@pytest.mark.parametrize("field,value", [("description", "  "), ("tool_call_id", ""), ("request_message_id", 0)])
def test_draft_rejects_missing_required_data(field, value):
    from dataclasses import asdict
    values = asdict(_draft())
    values.pop("confirmation_id")
    values[field] = value
    with pytest.raises(ValueError):
        TicketDraft.create(**values)


def test_confirmation_rejects_context_identity_mismatch(monkeypatch):
    monkeypatch.setattr("app.tools.ticket_confirmation.interrupt", lambda _: pytest.fail("must reject before interrupt"))
    with pytest.raises(PermissionError):
        confirm_ticket(_draft(), ExecutionContext(2, "another-call"))


def test_model_arguments_cannot_supply_request_evidence():
    with pytest.raises(PermissionError):
        draft_from_call({"ticket_request": {"status": "active", "message_id": 10}}, ExecutionContext(1, "call-1"))


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["cancel", "confirm", "tampered"])
async def test_real_graph_interrupt_gates_engine_writes_and_audit(decision):
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from app.tools.contracts import ToolDefinition, ToolOutcome
    from app.tools.executor import ToolEngine
    from app.tools.ticket_confirmation import confirm_ticket_call
    from types import SimpleNamespace

    writes, audits = [], []

    async def invoke(args, context):
        writes.append(args)
        return ToolOutcome("success")

    async def audit(record):
        audits.append(record)

    definition = ToolDefinition("create_ticket", "", {"type": "object"}, "builtin", None, None, invoke, effect="write")
    engine = ToolEngine(SimpleNamespace(load=lambda: SimpleNamespace(permits=lambda _: True)), audit, 1, 2)
    context = ExecutionContext(1, "call-1", ticket_request={"status": "active", "message_id": 10}, confirmation=confirm_ticket_call)
    args = {"ticket_type": "售后", "description": "猫粮包装破损"}

    async def node(state):
        message = await engine.execute({"id": "call-1", "name": "create_ticket", "args": args}, context, {"create_ticket": definition})
        return {"category": message.artifact["category"]}

    builder = StateGraph(dict)
    builder.add_node("ticket", node)
    builder.add_edge(START, "ticket")
    builder.add_edge("ticket", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "confirmation-test"}}
    paused = await graph.ainvoke({}, config)
    payload = paused["__interrupt__"][0].value
    assert writes == [] and audits == []
    resumed = await graph.ainvoke(Command(resume={
        "decision": "confirm" if decision == "tampered" else decision,
        "confirmation_id": "wrong" if decision == "tampered" else payload["confirmation_id"],
    }), config)
    assert len(writes) == (1 if decision == "confirm" else 0)
    assert len(audits) == 1
    assert resumed["category"] == ("success" if decision == "confirm" else "permission_denied")
