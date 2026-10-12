import hashlib

import pytest

from app.db.models import VectorizeStatus
from app.db.repository import KnowledgeRepository
from app.kb.chunking import Chunk
from app.kb.dualwrite import vectorize_pending, vectorize_text, write_chunks_pending
from app.kb.milvus_client import MilvusKnowledgeClient

DIM = 16


def _chunk(q, a, category="政策"):
    return Chunk(
        category=category, questions=q, answer=a, section_path="政策 > 运费", content_type="policy"
    )


class FakeEmbed:
    """确定性伪向量 + 可注入失败次数（便于断点续跑断言）。"""

    def __init__(self, dim=DIM, fail_after=None):
        self._dim = dim
        self._fail_after = fail_after
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        if self._fail_after is not None and self.calls > self._fail_after:
            raise RuntimeError("模拟建库中断")
        return [self._vec(t) for t in texts]

    def _vec(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) - 0.5 for i in range(self._dim)]


@pytest.fixture
def milvus(tmp_path):
    m = MilvusKnowledgeClient(uri=str(tmp_path / "kb.db"), collection="knowledge", dim=DIM)
    m.ensure_collection()
    yield m
    m.close()


def test_vectorize_text_is_category_questions_answer():
    assert vectorize_text("甲", "问", "答") == "甲\n问\n答"


async def test_write_then_vectorize(session_factory, milvus):
    ids = await write_chunks_pending(
        session_factory, [_chunk("邮费是多少", "满 99 包邮"), _chunk("退货政策", "7 天无理由")]
    )
    assert len(ids) == 2
    async with session_factory() as s:
        assert await KnowledgeRepository(s).count_by_status(VectorizeStatus.PENDING) == 2

    done = await vectorize_pending(session_factory, embed=FakeEmbed(), milvus=milvus)
    assert done == 2
    assert milvus.count() == 2
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        assert await repo.count_by_status(VectorizeStatus.PENDING) == 0
        assert await repo.count_by_status(VectorizeStatus.DONE) == 2


async def test_interrupted_vectorize_is_resumed_on_rerun(session_factory, milvus):
    """验收②：中途失败 → 重跑只捞 pending 补齐，无重无漏。"""
    await write_chunks_pending(
        session_factory, [_chunk("一", "a1"), _chunk("二", "a2"), _chunk("三", "a3")]
    )

    broken = FakeEmbed(fail_after=1)  # 第一批成功后、第二批失败
    with pytest.raises(RuntimeError):
        await vectorize_pending(session_factory, embed=broken, milvus=milvus, batch_size=1)

    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        assert await repo.count_by_status(VectorizeStatus.PENDING) > 0
        assert await repo.count_by_status(VectorizeStatus.DONE) >= 1

    resumed = await vectorize_pending(session_factory, embed=FakeEmbed(), milvus=milvus, batch_size=1)
    assert resumed == 2

    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        assert await repo.count_by_status(VectorizeStatus.PENDING) == 0
        assert await repo.count_by_status(VectorizeStatus.DONE) == 3
    assert milvus.count() == 3
