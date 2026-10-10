from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.context_service import ContextStore


def test_append_and_get():
    store = ContextStore()
    store.append("s1", HumanMessage(content="你好"))
    store.append("s1", AIMessage(content="你好，请问有什么可以帮您？"))
    assert len(store.get_messages("s1")) == 2
    assert store.get_messages("nope") == []


def test_count_tokens_monotonic():
    store = ContextStore()
    short = store.count_tokens([HumanMessage(content="你好")])
    long = store.count_tokens([HumanMessage(content="你好" * 100)])
    assert short > 0
    assert long > short


def test_trim_drops_oldest_keeps_newest():
    store = ContextStore(token_budget=60)
    system = SystemMessage(content="你是客服")
    for i in range(10):
        store.append("s1", AIMessage(content="这是第" + str(i) + "条很长的历史消息内容"))
    store.append("s1", HumanMessage(content="最新问题"))
    history = store.trim_history("s1", system)
    # 最新一条用户消息必须保留
    assert history[-1].content == "最新问题"
    # system + history 不超预算
    assert store.count_tokens([system] + history) <= 60
