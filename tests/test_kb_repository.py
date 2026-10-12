from dataclasses import dataclass

from sqlalchemy import select, text

from app.db.models import KnowledgeChunk, StagingStatus, VectorizeStatus
from app.db.repository import KnowledgeRepository


@dataclass
class FakeChunk:
    category: str
    questions: str
    answer: str
    section_path: str | None = None
    content_type: str = "md"
    is_key_clause: bool = False


def chunk(q: str, *, path: str = "文档 > 甲", answer: str = "答案正文") -> FakeChunk:
    return FakeChunk(category="文档", questions=q, answer=answer, section_path=path)


async def test_insert_returns_ids_and_pending(session_factory):
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        ids = await repo.insert_chunks([chunk("问一"), chunk("问二")])
        await s.commit()
    assert len(ids) == 2
    async with session_factory() as s:
        assert await KnowledgeRepository(s).count_by_status(VectorizeStatus.PENDING) == 2


async def test_link_neighbors(session_factory):
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        ids = await repo.insert_chunks([chunk("甲"), chunk("乙"), chunk("丙")])
        await repo.link_neighbors(ids)
        await s.commit()
    async with session_factory() as s:
        rows = {r.id: r for r in await KnowledgeRepository(s).get_by_ids(ids)}
    assert rows[ids[0]].prev_chunk_id is None and rows[ids[0]].next_chunk_id == ids[1]
    assert rows[ids[1]].prev_chunk_id == ids[0] and rows[ids[1]].next_chunk_id == ids[2]
    assert rows[ids[2]].next_chunk_id is None


async def test_doc_root_exists_and_delete(session_factory):
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        await repo.insert_chunks([chunk("q", path="退货与运费政策 > 运费")])
        await repo.insert_chunks([chunk("q2", path="商品常见问题 > 价格")])
        await repo.insert_chunks([chunk("q3", path="退货与运费政策补充 > 甲")])
        await s.commit()
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        assert await repo.doc_root_exists("退货与运费政策") is True
        assert await repo.doc_root_exists("不存在") is False
        deleted = await repo.delete_by_doc_root("退货与运费政策")
        await s.commit()
    assert len(deleted) == 1
    async with session_factory() as s:
        left = (await s.execute(select(KnowledgeChunk.section_path))).scalars().all()
    assert set(left) == {"商品常见问题 > 价格", "退货与运费政策补充 > 甲"}


async def test_list_pending_and_mark_vectorized(session_factory):
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        ids = await repo.insert_chunks([chunk("甲"), chunk("乙")])
        await repo.mark_vectorized(ids[0], str(ids[0]))
        await s.commit()
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        pending = await repo.list_pending()
        assert [r.id for r in pending] == [ids[1]]
        assert await repo.count_by_status(VectorizeStatus.DONE) == 1


async def test_get_by_ids_preserves_input_order(session_factory):
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        ids = await repo.insert_chunks([chunk("甲"), chunk("乙"), chunk("丙")])
        await s.commit()
    async with session_factory() as s:
        rows = await KnowledgeRepository(s).get_by_ids([ids[2], ids[0]])
    assert [r.id for r in rows] == [ids[2], ids[0]]


async def test_list_existing_qa_includes_multiline_questions(session_factory):
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        await repo.insert_chunks([chunk("邮费是多少\n运费怎么算", answer="满 99 包邮")])
        await s.commit()
    async with session_factory() as s:
        pairs = await KnowledgeRepository(s).list_existing_qa()
    assert pairs == [("邮费是多少\n运费怎么算", "满 99 包邮")]


async def test_staging_lifecycle(session_factory):
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        await repo.add_staging("b1", "conversation:1", "能开发票吗", "可以")
        await repo.add_staging("b1", "conversation:1", "发什么快递", "默认中通")
        await s.commit()
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        rows = await repo.list_staging("b1", StagingStatus.EXTRACTED)
        assert len(rows) == 2
        await repo.set_staging_status([rows[0].id], StagingStatus.DISCARDED)
        await s.commit()
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        assert len(await repo.list_staging("b1", StagingStatus.EXTRACTED)) == 1
        assert len(await repo.list_staging("b1", StagingStatus.DISCARDED)) == 1
        await repo.clear_batch("b1")
        await s.commit()
    async with session_factory() as s:
        total = (await s.execute(text("SELECT COUNT(*) FROM qa_extraction_staging"))).scalar()
    assert total == 0
