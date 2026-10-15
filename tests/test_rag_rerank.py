import json

import pytest

from app.config import Settings
from app.rag.rerank import RerankClient, RerankError, build_rerank_client


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _client(monkeypatch, payload, captured=None):
    def fake_urlopen(req, timeout=None):
        if captured is not None:
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["auth"] = req.headers.get("Authorization")
            captured["timeout"] = timeout
        return FakeResponse(payload)

    monkeypatch.setattr("app.rag.rerank.urllib.request.urlopen", fake_urlopen)
    return RerankClient(
        base_url="https://api.example.invalid/v1", api_key="sk-secret", model="m", timeout=7.0
    )


def test_rerank_returns_sorted_index_score_pairs(monkeypatch):
    captured = {}
    client = _client(monkeypatch, {"results": [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.3},
    ]}, captured)
    out = client.rerank("邮费是多少", ["甲", "乙", "丙"], top_n=2)
    assert out == [(2, 0.9), (0, 0.3)]
    assert captured["url"] == "https://api.example.invalid/v1/rerank"
    assert captured["body"] == {
        "model": "m", "query": "邮费是多少", "documents": ["甲", "乙", "丙"], "top_n": 2,
    }
    assert captured["timeout"] == 7.0


def test_rerank_never_logs_or_returns_the_key(monkeypatch, caplog):
    # R13 订正：原稿用 documents=[] + {"results": []}，但空候选按设计直接 return []（见
    # test_rerank_empty_documents_short_circuits），且 {"results": []} 本身也不触发异常 →
    # pytest.raises 必挂。改成真正走异常路径：缺 "results" 键 → KeyError → RerankError。
    client = _client(monkeypatch, {})
    with pytest.raises(RerankError) as excinfo:
        client.rerank("q", ["甲"], top_n=1)
    assert "sk-secret" not in caplog.text
    assert "sk-secret" not in str(excinfo.value)


def test_rerank_http_error_raises_rerank_error(monkeypatch):
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("app.rag.rerank.urllib.request.urlopen", boom)
    client = RerankClient(base_url="https://x.invalid/v1", api_key="k", model="m")
    with pytest.raises(RerankError):
        client.rerank("q", ["a"], top_n=1)


def test_rerank_malformed_body_raises_rerank_error(monkeypatch):
    client = _client(monkeypatch, {"results": [{"nope": 1}]})
    with pytest.raises(RerankError):
        client.rerank("q", ["a"], top_n=1)


def test_rerank_empty_documents_short_circuits(monkeypatch):
    def boom(req, timeout=None):  # 不应被调用
        raise AssertionError("空候选不该打上游")

    monkeypatch.setattr("app.rag.rerank.urllib.request.urlopen", boom)
    client = RerankClient(base_url="https://x.invalid/v1", api_key="k", model="m")
    assert client.rerank("q", [], top_n=5) == []


def test_build_rerank_client_reuses_embed_credentials_when_blank():
    s = Settings(
        _env_file=None, embed_base_url="https://embed.invalid/v1",
        embed_api_key="sk-embed", rerank_model="BAAI/bge-reranker-v2-m3",
    )
    client = build_rerank_client(s)
    assert client.is_configured() is True


def test_build_rerank_client_without_any_key_is_not_configured():
    s = Settings(_env_file=None, embed_api_key="", rerank_api_key="")
    assert build_rerank_client(s).is_configured() is False
