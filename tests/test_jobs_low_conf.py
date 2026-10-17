from app.db.models import LowConfidenceSource
from app.db.repository import LowConfidenceRepository
from app.jobs.low_conf import format_counts, format_questions, parse_args


def test_parse_args_defaults():
    args = parse_args([])
    assert args.count is False
    assert args.source is None
    assert args.limit == 20


def test_parse_args_source_and_limit():
    args = parse_args(["--source", "self_check", "--limit", "5"])
    assert args.source == "self_check"
    assert args.limit == 5


def test_format_questions_shows_raw_words_source_and_reason():
    class Row:
        id = 1
        created_at = None
        source = LowConfidenceSource.RETRIEVAL_LOW_CONF
        raw_question = "今天天气怎么样"
        reason = "召回结果为空"

    text = format_questions([Row()])
    assert "今天天气怎么样" in text
    assert "retrieval_low_conf" in text
    assert "召回结果为空" in text


def test_format_questions_empty_pool():
    assert "池中暂无" in format_questions([])


def test_format_counts_lists_every_source():
    text = format_counts({"retrieval_low_conf": 2, "self_check": 1})
    assert "retrieval_low_conf: 2" in text
    assert "self_check: 1" in text


async def test_main_list_reads_from_db(session_factory, monkeypatch, capsys):
    from app.jobs import low_conf

    async with session_factory() as session:
        await LowConfidenceRepository(session).add(
            None, "帮我订火车票", LowConfidenceSource.RETRIEVAL_LOW_CONF, "召回结果为空"
        )
        await session.commit()

    monkeypatch.setattr(low_conf, "SessionFactory", session_factory)
    await low_conf.run(parse_args(["--list"]))
    out = capsys.readouterr().out
    assert "帮我订火车票" in out
