from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.services.context_manager import (
    Turn,
    classify_turn,
    render_layer2_turn,
    serialize_message,
    split_turns,
)


def _user(msg_id: int, text: str) -> HumanMessage:
    return HumanMessage(content=text, id=f"db:{msg_id}")


def test_split_turns_by_db_anchor_and_keeps_tool_pairing():
    messages = [
        _user(1, "问1"),
        AIMessage(content="答1"),
        _user(2, "问2"),
        AIMessage(content="", tool_calls=[{
            "name": "query_order", "args": {"order_id": "1001"},
            "id": "call_2", "type": "tool_call",
        }]),
        ToolMessage(content="订单1001已签收", tool_call_id="call_2", name="query_order"),
        AIMessage(content="答2"),
        _user(3, "问3"),
    ]
    turns = split_turns(messages)
    assert [t.first_user_msg_id for t in turns] == [1, 2, 3]
    assert [len(t.messages) for t in turns] == [2, 4, 1]
    assert isinstance(turns[1].messages[2], ToolMessage)


def test_legacy_human_without_db_id_joins_next_anchored_turn():
    messages = [
        HumanMessage(content="旧问"),
        AIMessage(content="旧答"),
        _user(5, "新问"),
        AIMessage(content="新答"),
    ]
    turns = split_turns(messages)
    assert len(turns) == 1
    assert turns[0].first_user_msg_id == 5
    assert [m.content for m in turns[0].messages] == ["旧问", "旧答", "新问", "新答"]


def test_layer_classification_is_inclusive_at_first_user_message():
    t1 = Turn(1, [_user(1, "a")])
    t2 = Turn(2, [_user(2, "b")])
    t3 = Turn(3, [_user(3, "c")])
    assert classify_turn(t1, 1, 3) == 3
    assert classify_turn(t2, 1, 3) == 2
    assert classify_turn(t3, 1, 3) == 1
    assert classify_turn(Turn(None, []), 1, 3) == 1


def test_layer2_keeps_user_text_truncates_assistant_and_hides_tool_result():
    long_text = "客" * 100
    turn = Turn(1, [
        _user(1, "用户原话"),
        AIMessage(content="", tool_calls=[{
            "name": "query_order", "args": {"order_id": "1001"},
            "id": "c1", "type": "tool_call",
        }]),
        ToolMessage(content="很长" * 100, tool_call_id="c1", name="query_order"),
        AIMessage(content=long_text),
    ])
    rendered = render_layer2_turn(turn, prefix_chars=60)
    assert rendered[0].content == "用户原话"
    assert len(rendered[3].content) == 61
    assert rendered[3].content.endswith("…")
    assert rendered[1].tool_calls[0]["id"] == "c1"
    assert rendered[2].tool_call_id == "c1"
    assert rendered[2].content == "[工具结果已省略：query_order c1]"


def test_serialize_message_keeps_protocol_fields():
    msg = ToolMessage(content="结果", tool_call_id="c1", name="query_order", id="db:9")
    payload = serialize_message(msg)
    assert payload["role"] == "tool"
    assert payload["tool_call_id"] == "c1"
    assert payload["name"] == "query_order"
    assert payload["id"] == "db:9"
