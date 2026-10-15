import pytest

from app.prompts.rag import RewriteOutput
from app.rag.rewrite import RewriteResult, rewrite_query


class FakeModel:
    def __init__(self, result=None, boom=None):
        self._result = result
        self._boom = boom
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self._boom is not None:
            raise self._boom
        return self._result


def test_rewrite_returns_standard_query_and_synonyms():
    model = FakeModel(RewriteOutput(standard_query="运费怎么算", synonyms=["邮费", "快递费"]))
    out = rewrite_query("买满多少不用掏邮费啊", model=model)
    assert out == RewriteResult(standard_query="运费怎么算", synonyms=["邮费", "快递费"], degraded=False)
    assert model.calls == 1


def test_rewrite_falls_back_to_raw_query_on_error():
    """改写是增益不是依赖：失败就退回原话，不阻断链路（spec §7.2）。"""
    model = FakeModel(boom=RuntimeError("上游 500"))
    out = rewrite_query("买满多少不用掏邮费啊", model=model)
    assert out.standard_query == "买满多少不用掏邮费啊"
    assert out.synonyms == []
    assert out.degraded is True


def test_rewrite_falls_back_when_standard_query_is_blank():
    model = FakeModel(RewriteOutput(standard_query="   ", synonyms=["甲"]))
    out = rewrite_query("原话", model=model)
    assert out.standard_query == "原话"
    assert out.degraded is True


def test_rewrite_dedupes_and_caps_synonyms():
    model = FakeModel(RewriteOutput(
        standard_query="标准问法",
        synonyms=["运费", "运费", " 邮费 ", "", "快递费", "a", "b", "c"],
    ))
    out = rewrite_query("原话", model=model)
    assert out.synonyms == ["运费", "邮费", "快递费", "a", "b"]
    assert out.degraded is False


def test_build_query_for_bm25_includes_raw_standard_and_synonyms():
    from app.rag.rewrite import build_bm25_query

    result = RewriteResult(standard_query="运费怎么算", synonyms=["邮费"], degraded=False)
    assert build_bm25_query("买满多少不用掏邮费啊", result) == "买满多少不用掏邮费啊 运费怎么算 邮费"


def test_build_bm25_query_dedupes_repeated_terms():
    from app.rag.rewrite import build_bm25_query

    result = RewriteResult(standard_query="运费怎么算", synonyms=["运费怎么算"], degraded=False)
    assert build_bm25_query("运费怎么算", result) == "运费怎么算"
