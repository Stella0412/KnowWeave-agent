from langgraph.checkpoint.memory import InMemorySaver

from app.graph.build import build_graph


def test_build_graph_accepts_callbacks_and_keeps_state_api():
    graph = build_graph(checkpointer=InMemorySaver(), callbacks=[object()])
    assert hasattr(graph, "aget_state")
    assert hasattr(graph, "aupdate_state")
    assert hasattr(graph, "ainvoke")
    assert hasattr(graph, "astream")
