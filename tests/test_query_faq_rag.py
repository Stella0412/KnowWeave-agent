import app.tools.business as business
from app.core.retrieval import NOT_FOUND, UNAVAILABLE, RetrievalHit


async def test_query_faq_hit_formats_and_keeps_contract(monkeypatch):
    hits = [RetrievalHit(1, "邮费是多少\n运费怎么算", "满 99 包邮", 0.9)]
    monkeypatch.setattr(business, "search_knowledge", lambda q, k: hits)
    tool = business.build_query_faq(session_factory=None)
    out = await tool.ainvoke({"keyword": "邮费是多少"})
    assert out == "问：邮费是多少 / 运费怎么算\n答：满 99 包邮"
    assert list(tool.args_schema.model_fields) == ["keyword"]


async def test_query_faq_miss_returns_not_found(monkeypatch):
    monkeypatch.setattr(business, "search_knowledge", lambda q, k: [])
    tool = business.build_query_faq(session_factory=None)
    assert await tool.ainvoke({"keyword": "今天天气"}) == NOT_FOUND


async def test_query_faq_does_not_raise_on_search_failure(monkeypatch):
    def boom(q, k):
        raise RuntimeError("milvus 连不上")

    monkeypatch.setattr(business, "search_knowledge", boom)
    tool = business.build_query_faq(session_factory=None)
    assert await tool.ainvoke({"keyword": "邮费是多少"}) == UNAVAILABLE
