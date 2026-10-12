from sqlalchemy import select

from app.db.models import Conversation, KnowledgeChunk, Message, MessageRole, StagingStatus
from app.db.repository import KnowledgeRepository
from app.kb.mining import QABatch, QAMiner, QAPair

CONV_A = [
    ("user", "你们发什么快递呀"),
    ("assistant", "默认中通快递。"),
]
CONV_B = [
    ("user", "能开发票吗"),
    ("assistant", "可以开发票，下单备注抬头。"),
]


class StubModel:
    """按调用次数依次返回预设的 QABatch。"""

    def __init__(self, batches):
        self._batches = list(batches)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return self._batches.pop(0)


async def _seed_convs(session_factory, convs):
    async with session_factory() as s:
        for turns in convs:
            conv = Conversation(user_id="anonymous")
            s.add(conv)
            await s.flush()
            for role, content in turns:
                s.add(
                    Message(
                        conversation_id=conv.id,
                        role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
                        content=content,
                    )
                )
        await s.commit()


async def test_mine_extracts_keeps_and_writes_chunks(session_factory):
    await _seed_convs(session_factory, [CONV_A, CONV_B])
    model = StubModel(
        [
            QABatch(pairs=[QAPair(question="你们发什么快递", answer="默认中通快递")]),
            QABatch(pairs=[QAPair(question="能开发票吗", answer="可以，下单备注抬头")]),
        ]
    )
    report = await QAMiner(session_factory, model, batch_size=1).mine(run_id="mine-test")
    assert report.conversations == 2 and report.batches == 2
    assert report.extracted == 2 and report.kept == 2 and report.discarded == 0

    async with session_factory() as s:
        rows = await KnowledgeRepository(s).list_pending()
    assert [r.content_type for r in rows] == ["mined", "mined"]
    assert rows[0].section_path is None
    assert rows[0].category == "对话挖掘"


async def test_mine_dedupes_within_batch(session_factory):
    await _seed_convs(session_factory, [CONV_A])
    model = StubModel(
        [
            QABatch(
                pairs=[
                    QAPair(question="能开发票吗", answer="可以，下单备注抬头"),
                    QAPair(question="能开发票吗！", answer="可以，下单备注抬头。"),
                ]
            )
        ]
    )
    report = await QAMiner(session_factory, model, batch_size=1).mine(run_id="mine-test")
    assert report.extracted == 2 and report.kept == 1 and report.discarded == 1


async def test_mine_rerun_produces_no_duplicates(session_factory):
    """幂等：同 run_id 重跑 → 已入库的问答对全被去重丢弃，knowledge_chunks 不新增。"""
    await _seed_convs(session_factory, [CONV_A, CONV_B])
    batch = [
        QABatch(pairs=[QAPair(question="你们发什么快递", answer="默认中通")]),
        QABatch(pairs=[QAPair(question="能开发票吗", answer="可以")]),
    ]
    first = await QAMiner(session_factory, StubModel(batch), batch_size=1).mine(run_id="mine-rerun")
    assert first.kept == 2

    second = await QAMiner(session_factory, StubModel(list(batch)), batch_size=1).mine(
        run_id="mine-rerun"
    )
    assert second.extracted == 2 and second.kept == 0 and second.discarded == 2

    async with session_factory() as s:
        total = len((await s.execute(select(KnowledgeChunk))).scalars().all())
    assert total == 2


async def test_mine_skips_conversations_without_user_assistant_pairs(session_factory):
    await _seed_convs(session_factory, [[("user", "在吗")]])
    report = await QAMiner(session_factory, StubModel([]), batch_size=5).mine(run_id="mine-empty")
    assert report.conversations == 0 and report.batches == 0 and report.kept == 0


async def test_mine_rolls_back_batch_on_model_failure(session_factory):
    await _seed_convs(session_factory, [CONV_A, CONV_B])

    class Boom:
        async def ainvoke(self, messages):
            raise RuntimeError("LLM 超时")

    report = await QAMiner(session_factory, Boom(), batch_size=1).mine(run_id="mine-boom")
    assert report.extracted == 0
    async with session_factory() as s:
        rows = await KnowledgeRepository(s).list_staging(None, StagingStatus.EXTRACTED)
    assert rows == []
