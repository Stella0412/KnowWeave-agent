from sqlalchemy import select

from app.db.models import LowConfidenceQuestion, LowConfidenceStatus
from app.db.repository import LowConfidenceRepository
from app.services.flywheel_service import FlywheelService


async def _root(session_factory, question, *, count=1, status=LowConfidenceStatus.PENDING,
                chunk=None, fp=None):
    async with session_factory() as session:
        row = await LowConfidenceRepository(session).add_question(
            conversation_id=1, raw_question=question, source="self_check",
            reason="r", message_id=None, fingerprint=fp if fp is not None else "",
            merged_into_id=None, duplicate_count=count, status=status,
            citations=None, knowledge_chunk_id=chunk, resolution=None,
        )
        await session.commit()
        return row.id


async def test_backfill_and_exact_merge(session_factory):
    a = await _root(session_factory, "保价规则是什么", count=2, chunk=7)
    b = await _root(session_factory, "保价规则 是什么", count=3)
    service = FlywheelService(session_factory)
    assert await service.backfill_fingerprints() == 2
    report = await service.merge_exact()
    assert report["exact_merged"] == 1
    assert report["clusters"] == 1
    async with session_factory() as session:
        loser = await session.get(LowConfidenceQuestion, b)
        canonical = await session.get(LowConfidenceQuestion, a)
    assert loser.status is LowConfidenceStatus.MERGED
    assert loser.merged_into_id == a
    assert loser.duplicate_count == 1
    assert canonical.duplicate_count == 5
    assert canonical.knowledge_chunk_id == 7


async def test_exact_does_not_merge_cross_status(session_factory):
    await _root(session_factory, "保价规则是什么", status=LowConfidenceStatus.CLOSED)
    await _root(session_factory, "保价规则是什么", status=LowConfidenceStatus.PENDING)
    report = await FlywheelService(session_factory).merge_exact()
    assert report["exact_merged"] == 0
    assert report["skipped_cross_status"] == 1


async def test_exact_dry_run_writes_nothing(session_factory):
    await _root(session_factory, "保价规则是什么")
    await _root(session_factory, "保价规则是什么")
    service = FlywheelService(session_factory)
    await service.backfill_fingerprints()
    report = await service.merge_exact(dry_run=True)
    assert report["exact_merged"] == 1
    async with session_factory() as session:
        rows = (await session.execute(select(LowConfidenceQuestion))).scalars().all()
    assert all(row.status is LowConfidenceStatus.PENDING for row in rows)


async def test_semantic_merge_and_degraded(session_factory):
    await _root(session_factory, "保价怎么申请")
    await _root(session_factory, "价保如何申请")

    def embed(texts):
        return [[1.0, 0.0] for _ in texts]

    service = FlywheelService(session_factory, embed=embed)
    await service.backfill_fingerprints()
    report = await service.merge_semantic(threshold=0.9, limit=100, batch_size=10)
    assert report["semantic_merged"] == 1
    assert report["degraded"] is False

    def boom(texts):
        raise RuntimeError("embedding down")

    report = await FlywheelService(session_factory, embed=boom).merge_semantic(
        threshold=0.9, limit=100, batch_size=10
    )
    assert report["degraded"] is True


def test_eval_dataset_has_60_valid_pairs():
    from eval.eval_ch09_merge import load_pairs
    pairs = load_pairs("eval/ch09_merge_pairs.jsonl")
    assert len(pairs) == 60
    assert {p["label"] for p in pairs} == {"dup", "distinct"}
    assert sum(p["label"] == "dup" for p in pairs) == 30


def test_cli_defaults_none_and_resolve_from_settings():
    from app.config import Settings
    from app.jobs.low_conf_merge import _resolve, parse_args

    args = parse_args([])
    assert args.threshold is None
    assert args.limit is None
    assert args.batch_size is None

    s = Settings(_env_file=None)
    assert _resolve(args, s) == (0.92, 2000, 64)

    args = parse_args(["--threshold", "0.95", "--limit", "10", "--batch-size", "5"])
    assert _resolve(args, s) == (0.95, 10, 5)


async def test_cli_exit_code_one_when_flags_conflict():
    from app.jobs.low_conf_merge import parse_args, run

    assert await run(parse_args(["--skip-semantic", "--require-semantic"])) == 1
