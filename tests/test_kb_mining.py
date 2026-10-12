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


async def test_mine_daily_rerun_with_new_run_id_does_not_duplicate(session_factory):
    """每日重跑（run_id 按日期变，如 mine-20261012 → mine-20261013）不得把上一轮的行再灌一遍。

    `clear_batch` 只清**本 run 的批次号**，上一轮 run 的 KEPT 行原样留在暂存表；而
    `_promote_kept` 捞的是「全部 KEPT 行」→ 每跑一次 knowledge_chunks（及 Milvus）就
    多一份，同一条问答在检索结果里重复出现。
    """
    await _seed_convs(session_factory, [CONV_A, CONV_B])
    pairs = [
        QABatch(pairs=[QAPair(question="你们发什么快递", answer="默认中通快递")]),
        QABatch(pairs=[QAPair(question="能开发票吗", answer="可以，下单备注抬头")]),
    ]
    first = await QAMiner(session_factory, StubModel(list(pairs)), batch_size=1).mine(
        run_id="mine-20261012"
    )
    assert first.kept == 2

    second = await QAMiner(session_factory, StubModel(list(pairs)), batch_size=1).mine(
        run_id="mine-20261013"
    )
    assert second.kept == 0

    async with session_factory() as s:
        total = len((await s.execute(select(KnowledgeChunk))).scalars().all())
        still_kept = await KnowledgeRepository(s).list_staging(None, StagingStatus.KEPT)
    assert total == 2
    assert still_kept == []


async def test_mine_promotes_row_left_kept_by_crashed_run(session_factory):
    """上一轮「已判定 KEEP、但还没入库」就崩了的行，本轮要补灌（而不是永远卡在 KEPT）。"""
    await _seed_convs(session_factory, [CONV_A])
    async with session_factory() as s:
        await KnowledgeRepository(s).add_staging(
            "mine-crashed-001", "conversation:1", "你们发什么快递", "默认中通快递"
        )
        await KnowledgeRepository(s).set_staging_status(
            [row.id for row in await KnowledgeRepository(s).list_staging(None, StagingStatus.EXTRACTED)],
            StagingStatus.KEPT,
        )
        await s.commit()

    report = await QAMiner(session_factory, StubModel([QABatch(pairs=[])]), batch_size=5).mine(run_id="mine-next-day")
    assert report.kept == 0  # 没有新增 KEEP，只是补灌上一轮的遗留行
    async with session_factory() as s:
        rows = (await s.execute(select(KnowledgeChunk))).scalars().all()
    assert [r.questions for r in rows] == ["你们发什么快递"]


async def test_mine_discards_assistant_non_answers(session_factory):
    """客服只答「暂时未查询到 / 帮您转人工」这类**非答复**，不能当知识入库。

    这是 MINING_SYSTEM_PROMPT 里「用户没得到明确答复…的不要抽」那句的确定性闸门：实测 LLM
    仍然会抽（本库真实数据：ch01/ch02 时期 6 条「暂时未查询到邮费」被挖进知识库，因为问句
    就是用户原话、彼此近乎一样，它们占满 top-5，把真正的运费政策文档挤到第 6 名 →「邮费是
    多少」答成「暂时未查询到」）。机器人自己的拒答变成知识再喂回自己，是自我强化的坏循环。
    """
    await _seed_convs(session_factory, [CONV_A])
    model = StubModel(
        [
            QABatch(
                pairs=[
                    QAPair(
                        question="邮费是多少",
                        answer="抱歉，我这边暂时未查询到具体的邮费信息，可以告诉我您所在的城市/地区？",
                    ),
                    QAPair(question="能开发票吗", answer="可以开发票，下单时备注抬头和税号即可。"),
                ]
            )
        ]
    )
    report = await QAMiner(session_factory, model, batch_size=1).mine(run_id="mine-guard")
    assert report.extracted == 2
    assert report.kept == 1 and report.non_answers == 1 and report.discarded == 0

    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        assert [r.questions for r in await repo.list_pending()] == ["能开发票吗"]
        dropped = await repo.list_staging(None, StagingStatus.DISCARDED)
    assert [r.question for r in dropped] == ["邮费是多少"]  # 仍在暂存表里留痕，不是静默丢掉


async def test_mine_drops_non_answer_left_kept_by_earlier_run(session_factory):
    """闸门必须也管**上一轮遗留的 KEPT 行** —— 否则只手删库里的假知识，下次重跑又原样灌回来。

    真实场景：ch01/ch02 时期那 6 条「暂时未查询到邮费」早已被判定 KEPT 并入库；加了闸门后
    若只对「本轮 EXTRACTED 行」生效，重跑时它们从 KEPT 分支直接进 `_promote_kept`，闸门形同虚设。
    """
    await _seed_convs(session_factory, [CONV_A])
    async with session_factory() as s:
        repo = KnowledgeRepository(s)
        await repo.add_staging("mine-old-001", "conversation:3", "邮费是多少", "暂时未查询到具体的邮费信息。")
        await repo.set_staging_status(
            [r.id for r in await repo.list_staging(None, StagingStatus.EXTRACTED)],
            StagingStatus.KEPT,
        )
        await s.commit()

    report = await QAMiner(session_factory, StubModel([QABatch(pairs=[])]), batch_size=5).mine(run_id="mine-next-day")
    assert report.kept == 0 and report.non_answers == 1
    async with session_factory() as s:
        assert (await s.execute(select(KnowledgeChunk))).scalars().all() == []
        assert await KnowledgeRepository(s).list_staging(None, StagingStatus.KEPT) == []


# 全部取自本库真实数据（ids 19/21/23/24/27/30/32/34/137/157），不是编的
REAL_NON_ANSWERS = [
    "暂时未查询到具体的邮费信息，可能是运费规则因地区、商品或活动而有所不同。可以告诉您所在的城市/地区。",
    "关于运费的具体标准，暂时未查询到相关信息，无法准确答复。可以告诉我具体商品或订单号。",
    "非常抱歉，关于运费的具体标准，我这边暂时未查询到相关信息，无法为您准确答复。",
    "目前系统中查询到的订单 1001 物流信息仅显示「已到达转运中心」，并未包含具体的城市/网点位置信息，因此我无法确认当前所在城市。",
    "这个我这边暂时无法提供准确信息，建议您联系人工客服。",
]
REAL_ANSWERS = [
    "可以开发票，下单时备注抬头和税号即可，电子发票 3 个工作日内发到邮箱。",
    "默认中通快递，偏远地区可能转其他快递。",
    "单笔订单实付满 99 元全国包邮（新疆、西藏、港澳台除外），不满 99 元收取运费 8 元。",
    "猫粮会用铝箔袋加纸箱双层包装，并在箱内加干燥剂。",
]


def test_is_non_answer_classifies_real_data():
    """闸门的判据来自真实库数据：10 条被挖进来的非答复必须全中，4 条正经答复必须全不中。"""
    from app.kb.mining import is_non_answer

    missed = [a for a in REAL_NON_ANSWERS if not is_non_answer(a)]
    false_alarm = [a for a in REAL_ANSWERS if is_non_answer(a)]
    assert missed == [] and false_alarm == []


async def test_mine_discards_orders_limitation_answer(session_factory):
    """「查到了但查不到城市，所以我无法确认」这类**订单局限**答复同样是非答复，不能入库。

    真实案例：`id=157` 就是这么进来的 —— 它不含「未查询到」，所以第一版闸门放过了它，
    而它是 21/32/137 三条被删掉的假知识的换皮版本。
    """
    await _seed_convs(session_factory, [CONV_A])
    model = StubModel(
        [
            QABatch(
                pairs=[
                    QAPair(
                        question="现在的订单位置位于哪个城市",
                        answer=REAL_NON_ANSWERS[3],
                    )
                ]
            )
        ]
    )
    report = await QAMiner(session_factory, model, batch_size=1).mine(run_id="mine-guard2")
    assert report.extracted == 1 and report.kept == 0 and report.non_answers == 1
    async with session_factory() as s:
        assert (await s.execute(select(KnowledgeChunk))).scalars().all() == []


async def test_mine_skips_conversations_without_user_assistant_pairs(session_factory):
    await _seed_convs(session_factory, [[("user", "在吗")]])
    report = await QAMiner(session_factory, StubModel([QABatch(pairs=[])]), batch_size=5).mine(run_id="mine-empty")
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
