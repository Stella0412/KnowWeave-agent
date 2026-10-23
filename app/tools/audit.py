"""ch08 工具调用审计写入器。

写入器使用 session_factory 独立开启 session/事务，绝不复用业务会话；
任何失败（建会话、insert、commit）只记 warning、绝不抛给调用方，
保证审计不阻断、不回滚工具执行结果。
"""

import logging

from app.db.models import AuditStatus, ToolAuditLog, ToolSource
from app.tools.contracts import AuditRecord

logger = logging.getLogger(__name__)

MAX_ERROR_MESSAGE = 512


async def write_audit(session_factory, record: AuditRecord) -> None:
    """尽力写入一条审计记录；失败只告警，不抛出异常。"""
    try:
        status = AuditStatus(record.status)
        source = ToolSource(record.tool_source)
        error_message = record.error_message
        if error_message is not None and len(error_message) > MAX_ERROR_MESSAGE:
            error_message = error_message[:MAX_ERROR_MESSAGE]
        session = session_factory()
    except Exception:
        # asyncio.CancelledError 属 BaseException，不会在此被吞掉
        logger.warning("write_audit aborted before opening a session", exc_info=True)
        return

    try:
        async with session:
            session.add(
                ToolAuditLog(
                    conversation_id=record.conversation_id,
                    tool_call_id=record.tool_call_id,
                    tool_name=record.tool_name,
                    tool_source=source,
                    mcp_server=record.mcp_server,
                    arguments=record.arguments,
                    result_summary=record.result_summary,
                    status=status,
                    error_message=error_message,
                    retry_count=record.retry_count,
                    duration_ms=record.duration_ms,
                )
            )
            await session.commit()
    except Exception:
        try:
            await session.rollback()
        except Exception:
            pass
        logger.warning("write_audit aborted during insert/commit", exc_info=True)
