import pytest
from sqlalchemy import select

from app.db.models import AuditStatus, ToolAuditLog, ToolSource
from app.tools.contracts import AuditRecord


def make_record(**overrides):
    fields = dict(
        conversation_id=7,
        tool_call_id="call-1",
        tool_name="demo",
        tool_source="builtin",
        mcp_server=None,
        arguments={"query": "订单"},
        result_summary="ok",
        status="成功",
        error_message=None,
        retry_count=0,
        duration_ms=12,
    )
    fields.update(overrides)
    return AuditRecord(**fields)


async def read_rows(session_factory):
    async with session_factory() as session:
        result = await session.execute(select(ToolAuditLog).order_by(ToolAuditLog.id))
        return result.scalars().all()


@pytest.mark.parametrize("status", ["成功", "失败", "超时", "校验拦下", "权限拒绝"])
async def test_each_status_inserts_and_reads_back(session_factory, status):
    from app.tools.audit import write_audit

    await write_audit(session_factory, make_record(status=status))

    rows = await read_rows(session_factory)
    assert len(rows) == 1
    assert rows[0].status is AuditStatus(status)
    assert rows[0].status.value == status


def test_conversation_id_is_nullable_and_has_no_foreign_key():
    column = ToolAuditLog.__table__.c.conversation_id
    assert column.nullable is True
    assert list(column.foreign_keys) == []
    assert list(ToolAuditLog.__table__.foreign_keys) == []


async def test_nullable_fields_are_persisted_as_none(session_factory):
    from app.tools.audit import write_audit

    await write_audit(session_factory, make_record(conversation_id=None, tool_call_id=None))

    row = (await read_rows(session_factory))[0]
    assert row.conversation_id is None
    assert row.tool_call_id is None


async def test_chinese_arguments_and_error_round_trip_unchanged(session_factory):
    from app.tools.audit import write_audit

    arguments = {"问题": "物流到哪里了", "明细": ["承运商：顺丰", {"备注": "签收"}]}
    error = "权限不足：需要管理员授权后重试"
    await write_audit(session_factory, make_record(
        tool_source="mcp", mcp_server="logistics",
        arguments=arguments, error_message=error, result_summary="结果：在途",
    ))

    row = (await read_rows(session_factory))[0]
    assert row.arguments == arguments
    assert row.error_message == error
    assert row.result_summary == "结果：在途"
    assert row.tool_source is ToolSource.MCP
    assert row.mcp_server == "logistics"


async def test_error_message_longer_than_512_is_truncated(session_factory):
    from app.tools.audit import write_audit

    await write_audit(session_factory, make_record(error_message="错" * 600))

    row = (await read_rows(session_factory))[0]
    assert len(row.error_message) == 512
    assert row.error_message == "错" * 512


async def test_audit_failure_is_isolated():
    from app.tools.audit import write_audit

    def broken_factory():
        raise OSError("database down")

    record = AuditRecord(None, "c1", "demo", "builtin", None, {}, "{}",
                         "成功", None, 0, 1)
    await write_audit(broken_factory, record)


async def test_insert_or_commit_failure_is_isolated():
    from app.tools.audit import write_audit

    class BrokenSession:
        def add(self, obj):
            pass

        async def __aenter__(self):
            raise RuntimeError("commit down")

        async def __aexit__(self, *exc):
            return False

        async def rollback(self):
            pass

    await write_audit(lambda: BrokenSession(), make_record())


async def test_write_audit_opens_an_independent_session(session_factory):
    from app.tools.audit import write_audit

    opened = {"n": 0}
    original = session_factory

    def counting_factory():
        opened["n"] += 1
        return original()

    await write_audit(counting_factory, make_record())

    assert opened["n"] == 1
    assert len(await read_rows(session_factory)) == 1
