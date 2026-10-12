"""双写落库：MySQL 权威源（pending）→ Milvus upsert → 回填 vector_id + done。

幂等/恢复：崩在任意批 → 重跑只捞 status='pending' → 按 id upsert（重写已 done 的同 id
无副作用）→ 直到全 done。满足验收②。
"""

import asyncio

from app.db.models import VectorizeStatus
from app.db.repository import KnowledgeRepository
from app.kb.chunking import Chunk


def vectorize_text(category: str, questions: str, answer: str) -> str:
    """拼向量化文本：严格 category + questions + answer 三格，元数据不进向量。"""
    return f"{category}\n{questions}\n{answer}"


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
        await asyncio.to_thread(
            milvus.upsert,
            [r.id for r in rows],
            vectors,
            [r.questions for r in rows],
            [r.answer for r in rows],
        )
        async with session_factory() as session:
            repo = KnowledgeRepository(session)
            for row in rows:
                await repo.mark_vectorized(row.id, str(row.id))
            await session.commit()
        total += len(rows)
    return total
