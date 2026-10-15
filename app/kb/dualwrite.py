"""双写落库：MySQL 权威源（pending）→ Milvus upsert → 回填 vector_id + done。

幂等/恢复：崩在任意批 → 重跑只捞 status='pending' → 按 id upsert（重写已 done 的同 id
无副作用）→ 直到全 done。满足验收②。
"""

import asyncio

from app.db.models import VectorizeStatus
from app.db.repository import KnowledgeRepository
from app.kb.chunking import Chunk
from app.kb.milvus_client import (
    ANSWER_MAX_LENGTH,
    META_MAX_LENGTH,
    QUESTION_MAX_LENGTH,
    ROW_CATEGORY_MAX_LENGTH,
    TEXT_MAX_LENGTH,
)


def vectorize_text(category: str, questions: str, answer: str) -> str:
    """拼向量化文本：严格 category + questions + answer 三格，元数据不进向量。"""
    return f"{category}\n{questions}\n{answer}"


def build_milvus_rows(chunk_rows: list, vectors: list[list[float]]) -> list[dict]:
    """把 MySQL 行 + 向量拼成 Milvus 整行。

    同一份 `vectorize_text` 既喂 dense 也喂 BM25：语义检索与字面检索看的是同一段文字，
    省掉「两路各自拼文本」的分叉。旧夹具的字段长度上限在这里统一截断，
    免得一条超长 chunk 让整批 upsert 失败。
    """
    rows = []
    for row, vector in zip(chunk_rows, vectors):
        rows.append({
            "id": int(row.id),
            "vector": vector,
            "text": vectorize_text(row.category, row.questions, row.answer)[:TEXT_MAX_LENGTH],
            "question": (row.questions or "")[:QUESTION_MAX_LENGTH],
            "answer": (row.answer or "")[:ANSWER_MAX_LENGTH],
            "category": (row.category or "")[:META_MAX_LENGTH],
            "content_type": (row.content_type or "md")[:META_MAX_LENGTH],
            "section_path": (row.section_path or "")[:META_MAX_LENGTH],
            "row_category": (getattr(row, "row_category", "") or "")[:ROW_CATEGORY_MAX_LENGTH],
            "is_key_clause": bool(row.is_key_clause),
        })
    return rows


async def write_chunks_pending(session_factory, chunks: list[Chunk]) -> list[int]:
    """阶段1：chunk 入 MySQL knowledge_chunks，状态 pending。"""
    async with session_factory() as session:
        ids = await KnowledgeRepository(session).insert_chunks(
            chunks, status=VectorizeStatus.PENDING
        )
        await session.commit()
    return ids


async def vectorize_pending(session_factory, *, embed, milvus, batch_size: int = 200) -> int:
    """阶段2：捞 pending → 拼文本 → embed → Milvus upsert（存 question/answer）→ 标 done。

    embed: callable(list[str]) -> list[list[float]]（即 embed_texts）
    """
    total = 0
    while True:
        async with session_factory() as session:
            rows = await KnowledgeRepository(session).list_pending(limit=batch_size)
        if not rows:
            break
        texts = [vectorize_text(r.category, r.questions, r.answer) for r in rows]
        vectors = embed(texts)
        await asyncio.to_thread(milvus.upsert, build_milvus_rows(rows, vectors))
        async with session_factory() as session:
            repo = KnowledgeRepository(session)
            for row in rows:
                await repo.mark_vectorized(row.id, str(row.id))
            await session.commit()
        total += len(rows)
    return total
