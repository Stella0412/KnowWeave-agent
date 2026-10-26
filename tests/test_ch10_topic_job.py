import pytest

from app.db.models import LowConfidenceSource, LowConfidenceStatus
from app.db.repository import LowConfidenceRepository, TopicRepository
from app.jobs.topic_classify import run_classify


class FakeClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    async def classify(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("service down")
        return {"model_version": "test", "results": [
            [{"label": "其他", "score": 0.9}] for _ in texts
        ]}


async def _seed(session_factory, count):
    async with session_factory() as session:
        repo = LowConfidenceRepository(session)
        for i in range(count):
            await repo.add_question(
                conversation_id=None, raw_question=f"问题{i}",
                source=LowConfidenceSource.RETRIEVAL_LOW_CONF, reason="test",
                message_id=None, fingerprint=f"fp{i}", merged_into_id=None,
                duplicate_count=1, status=LowConfidenceStatus.PENDING,
                citations=None, knowledge_chunk_id=None, resolution=None,
            )
        await session.commit()


async def test_below_batch_size_skips(session_factory):
    await _seed(session_factory, 3)
    result = await run_classify(session_factory, FakeClient(), batch_size=20)
    assert result["skipped"] is True
    assert result["classified"] == 0


async def test_writes_rows(session_factory):
    await _seed(session_factory, 25)
    client = FakeClient()
    result = await run_classify(session_factory, client, batch_size=20)
    assert result["classified"] == 20
    assert len(client.calls) == 1
    async with session_factory() as session:
        assert (await TopicRepository(session).coverage())["classified_roots"] == 20


async def test_service_failure_writes_nothing(session_factory):
    await _seed(session_factory, 25)
    with pytest.raises(RuntimeError):
        await run_classify(session_factory, FakeClient(fail=True), batch_size=20)
    async with session_factory() as session:
        assert (await TopicRepository(session).coverage())["classified_roots"] == 0


class BadLabelClient:
    async def classify(self, texts):
        return {"model_version": "bad", "results": [
            [{"label": "不存在的类目", "score": 0.9}] for _ in texts
        ]}


async def test_rejects_unknown_label_without_writing(session_factory):
    await _seed(session_factory, 25)
    with pytest.raises(ValueError):
        await run_classify(session_factory, BadLabelClient(), batch_size=20)
    async with session_factory() as session:
        assert (await TopicRepository(session).coverage())["classified_roots"] == 0


class CappedClient:
    def __init__(self, cap):
        self.cap = cap
        self.calls = []

    async def classify(self, texts):
        self.calls.append(len(texts))
        if len(texts) > self.cap:
            raise AssertionError(f"请求超过服务上限: {len(texts)} > {self.cap}")
        return {"model_version": "test", "results": [
            [{"label": "其他", "score": 0.9}] for _ in texts
        ]}


async def test_reclassify_all_chunks_requests_within_service_cap(session_factory):
    await _seed(session_factory, 130)
    client = CappedClient(128)
    result = await run_classify(session_factory, client, batch_size=20, reclassify_all=True)
    assert result["classified"] == 130
    assert client.calls and max(client.calls) <= 128
    async with session_factory() as session:
        assert (await TopicRepository(session).coverage())["classified_roots"] == 130


async def test_dry_run_does_not_write(session_factory):
    await _seed(session_factory, 25)
    result = await run_classify(session_factory, FakeClient(), batch_size=20, dry_run=True)
    assert result["dry_run"] is True
    assert result["selected"] == 20
    async with session_factory() as session:
        assert (await TopicRepository(session).coverage())["classified_roots"] == 0
