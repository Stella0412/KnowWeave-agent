from app.core.embeddings import build_embeddings, embed_texts


class FakeEmbeddings:
    def __init__(self, dim=1024):
        self.dim = dim
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[float(len(t))] * self.dim for t in texts]


def test_embed_texts_passes_through():
    fake = FakeEmbeddings(dim=8)
    vecs = embed_texts(["甲", "乙乙"], embeddings=fake)
    assert [len(v) for v in vecs] == [8, 8]
    assert vecs[0] == [1.0] * 8


def test_embed_texts_empty_skips_call():
    fake = FakeEmbeddings(dim=8)
    assert embed_texts([], embeddings=fake) == []
    assert fake.calls == 0


def test_build_embeddings_uses_settings():
    from app.config import Settings

    s = Settings(
        _env_file=None,
        embed_api_key="sk-test",
        embed_base_url="https://example.invalid/v1",
        embed_model="BAAI/bge-m3",
    )
    inner = build_embeddings(s)
    assert inner.model == "BAAI/bge-m3"
    assert inner.openai_api_base == "https://example.invalid/v1"
    # 第三方端点必须关掉 token 化输入（否则发 token 数组，端点不认）
    assert inner.check_embedding_ctx_length is False
    # 嵌入是检索链的硬依赖：一次 DNS 抖动不该让整轮 300 题评估作废
    assert inner.max_retries == 8
