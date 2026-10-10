from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.context_service import ContextStore


def test_count_tokens_monotonic():
    store = ContextStore()
    short = store.count_tokens([HumanMessage(content="你好")])
    long = store.count_tokens([HumanMessage(content="你好" * 100)])
    assert short > 0
    assert long > short


def test_trim_drops_oldest_keeps_newest():
    store = ContextStore(token_budget=60)
    system = SystemMessage(content="你是客服")
    history = [AIMessage(content="第" + str(i) + "条很长的历史消息内容") for i in range(10)]
    history.append(HumanMessage(content="最新问题"))
    trimmed = store.trim(system, history)
    # 最新一条用户消息必须保留
    assert trimmed[-1].content == "最新问题"
    # system + trimmed 不超预算
    assert store.count_tokens([system] + trimmed) <= 60


def test_trim_does_not_mutate_input():
    store = ContextStore(token_budget=1000)
    history = [HumanMessage(content="你好")]
    store.trim(SystemMessage(content="sys"), history)
    assert len(history) == 1
