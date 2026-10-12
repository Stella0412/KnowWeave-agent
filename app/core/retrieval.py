"""在线检索：query → embed → Milvus Top-K → 相似度阈值过滤。

检索直接从 Milvus 取回 question/answer（不回 MySQL）。这是 query_faq 的新内脏。
"""

from dataclasses import dataclass

from app.config import settings
from app.core.embeddings import embed_texts
from app.kb.milvus_client import MilvusKnowledgeClient, get_milvus_client

NOT_FOUND = "未找到相关常见问题"
UNAVAILABLE = "知识库暂不可用，请稍后再试"


@dataclass
class RetrievalHit:
    id: int
    question: str
    answer: str
    score: float


def search_knowledge(
    query: str,
    top_k: int,
    *,
    embeddings=None,
    milvus: MilvusKnowledgeClient | None = None,
    min_score: float | None = None,
) -> list[RetrievalHit]:
    """embed(query) → Milvus search → 阈值过滤。低于 min_score 算未命中（丢弃）。"""
    vector = embed_texts([query], embeddings=embeddings)[0]
    store = milvus or get_milvus_client()
    threshold = settings.retrieval_min_score if min_score is None else min_score
    hits = store.search(vector, top_k)
    return [
        RetrievalHit(id=h["id"], question=h["question"], answer=h["answer"], score=h["score"])
        for h in hits
        if h["score"] >= threshold
    ]


def format_hits(hits: list[RetrievalHit]) -> str:
    """把命中的 hit 拼成「问：… / 答：…」。多问法换行折叠成 " / "，保持一行一问。"""
    parts = []
    for h in hits:
        questions = " / ".join(line.strip() for line in h.question.splitlines() if line.strip())
        parts.append(f"问：{questions}\n答：{h.answer}")
    return "\n".join(parts)
