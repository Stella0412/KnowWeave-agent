from dataclasses import is_dataclass

from app.config import Settings
from app.graph.state import ConversationState, GraphContext
from app.rag.pipeline import TurnContext


def test_ch05_settings_defaults():
    s = Settings(llm_api_key="sk-test", _env_file=None)
    assert s.langgraph_checkpoint_path == "data/langgraph_checkpoints.sqlite"
    assert s.max_agent_steps == 6


def test_graph_context_carries_per_turn_dependencies():
    turn = TurnContext(raw_question="退货政策", conversation_id=7)
    ctx = GraphContext(
        model=object(),
        intent_model=object(),
        session_factory=object(),
        turn=turn,
    )
    assert is_dataclass(ctx)
    assert ctx.turn is turn
    assert ctx.token_budget == 4096
    assert ctx.tool_timeout == 5.0
    assert ctx.tool_retries == 1
    assert "messages" in ConversationState.__annotations__
