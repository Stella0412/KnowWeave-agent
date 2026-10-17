import pytest

from app.db.models import FaithCaseStatus
from app.db.repository import FaithCaseRepository
from app.jobs.faith_triage import format_cases, parse_args, run


def test_parse_args_list_defaults():
    args = parse_args(["--list"])
    assert args.list is True
    assert args.status is None


def test_parse_args_resolve_requires_eval_id():
    args = parse_args(["--resolve", "B03", "--status", "已解决", "--resolution", "补了规格条目"])
    assert args.resolve == "B03"
    assert args.status == "已解决"
    assert args.resolution == "补了规格条目"


def test_format_cases_shows_reason_and_seen_count():
    class Row:
        id = 1
        eval_id = "B03"
        bucket = "B_model"
        status = FaithCaseStatus.OPEN
        seen_count = 3
        query = "摄像头版是哪个型号"
        reason = "资料里只有 MH-FD30，答案却写了 MH-CAM1"
        last_seen_at = None
        resolution = None

    text = format_cases([Row()])
    assert "B03" in text and "B_model" in text and "3 次" in text
    assert "MH-CAM1" in text


def test_format_cases_empty():
    assert "暂无" in format_cases([])


async def _seed(session_factory, eval_id="B03"):
    async with session_factory() as session:
        await FaithCaseRepository(session).upsert_case(
            eval_id=eval_id, bucket="B_model", query="型号?", strategy="hybrid_rerank",
            answer="答", reason="编造", citations=[], judge_model="deepseek-chat",
        )
        await session.commit()


async def test_run_list_prints_open_cases(session_factory, capsys):
    await _seed(session_factory)
    await run(parse_args(["--list"]), session_factory=session_factory)
    assert "B03" in capsys.readouterr().out


async def test_run_resolve_marks_resolved(session_factory, capsys):
    await _seed(session_factory)
    code = await run(
        parse_args(["--resolve", "B03", "--status", "已解决", "--resolution", "补了规格条目"]),
        session_factory=session_factory,
    )
    assert code == 0
    async with session_factory() as session:
        case = (await FaithCaseRepository(session).list_cases())[0]
    assert case.status is FaithCaseStatus.RESOLVED
    assert case.resolution == "补了规格条目"


async def test_run_resolve_back_to_open_clears_resolution(session_factory):
    await _seed(session_factory)
    await run(parse_args(["--resolve", "B03", "--status", "已解决", "--resolution", "x"]),
              session_factory=session_factory)
    await run(parse_args(["--resolve", "B03", "--status", "未解决"]),
              session_factory=session_factory)
    async with session_factory() as session:
        case = (await FaithCaseRepository(session).list_cases())[0]
    assert case.status is FaithCaseStatus.OPEN
    assert case.resolution is None


async def test_run_resolve_unknown_eval_id_returns_nonzero(session_factory, capsys):
    code = await run(parse_args(["--resolve", "没有这题", "--status", "已解决"]),
                     session_factory=session_factory)
    assert code == 1
    assert "没有这题" in capsys.readouterr().err


async def test_run_resolve_requires_status(session_factory, capsys):
    with pytest.raises(SystemExit):
        parse_args(["--resolve", "B03"])
