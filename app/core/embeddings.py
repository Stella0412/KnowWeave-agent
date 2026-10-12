"""BGE-M3 嵌入封装：直连硅基流动 OpenAI 兼容端点 /v1/embeddings。"""

from langchain_openai import OpenAIEmbeddings

from app.config import Settings, settings

_embeddings: OpenAIEmbeddings | None = None


def build_embeddings(s: Settings | None = None) -> OpenAIEmbeddings:
    """注意 check_embedding_ctx_length 必须为 False：默认 True 会把输入 token 化后发数组，
    多数第三方 OpenAI 兼容端点不认（会 400）。"""
    s = s or settings
    return OpenAIEmbeddings(
        model=s.embed_model,
        base_url=s.embed_base_url,
        api_key=s.embed_api_key,
        check_embedding_ctx_length=False,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """进程级懒加载单例：避免每次检索都重建 httpx 客户端。"""
    global _embeddings
    if _embeddings is None:
        _embeddings = build_embeddings()
    return _embeddings


def embed_texts(
    texts: list[str], embeddings: OpenAIEmbeddings | None = None
) -> list[list[float]]:
    """把一批文本编码成向量。空列表直接返回空（不调用上游）。"""
    if not texts:
        return []
    emb = embeddings or get_embeddings()
    return emb.embed_documents(texts)
