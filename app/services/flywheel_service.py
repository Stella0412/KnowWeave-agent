"""ch09 低置信度池在线写入与飞轮服务。

在线只做三件事：算指纹、找根（含孤儿兜底）、命中只累加。绝不改根的业务状态。
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from app.config import settings as default_settings
from app.core.embeddings import embed_texts
from app.db.models import LowConfidenceQuestion, LowConfidenceStatus
from app.db.repository import LowConfidenceRepository
from app.kb.dedup import normalize_text
from app.kb.milvus_client import get_milvus_client
from app.rag.evidence import cap_citations

logger = logging.getLogger(__name__)
_MAX_HOPS = 10


def question_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecordResult:
    root_id: int
    occurrence_id: int
    created: bool
    duplicate_count: int
    status: str


class FlywheelService:
    def __init__(self, session_factory, *, settings=None,
                 embed=embed_texts, milvus_factory=get_milvus_client):
        self._session_factory = session_factory
        self._settings = settings or default_settings
        self._embed = embed
        self._milvus_factory = milvus_factory

    async def record_event(self, session, *, conversation_id, raw_question,
                           source, reason, message_id=None, citations=None):
        text = (raw_question or "").strip()
        if not text:
            raise ValueError("raw_question 不能为空")
        fp = question_fingerprint(text)
        repo = LowConfidenceRepository(session)
        root = await repo.get_root_for_update(fp)
        if root is None:
            candidate = await repo.get_any_for_update(fp)
            if candidate is not None:
                root = await self._resolve_locked_root(repo, candidate)
                if root is None:
                    logger.warning("池中同指纹行根链断裂/成环 fp=%s; 新建主问题", fp)
        now = datetime.now()
        # None = 这一轮没有检索快照；[] = 走了检索但没有可用片段。审核页要区分两者。
        snapshot = None if citations is None else cap_citations(citations)
        if root is None:
            row = await repo.add_question(
                conversation_id=conversation_id, raw_question=text,
                source=source, reason=reason, message_id=message_id,
                fingerprint=fp, merged_into_id=None, duplicate_count=1,
                status=LowConfidenceStatus.PENDING, citations=snapshot,
                knowledge_chunk_id=None, resolution=None, last_seen_at=now,
            )
            await session.flush()
            return RecordResult(row.id, row.id, True, 1,
                                LowConfidenceStatus.PENDING.value)
        root.duplicate_count = int(root.duplicate_count or 0) + 1
        root.last_seen_at = now
        occurrence = await repo.add_question(
            conversation_id=conversation_id, raw_question=text,
            source=source, reason=reason, message_id=message_id,
            fingerprint=fp, merged_into_id=root.id, duplicate_count=1,
            status=LowConfidenceStatus.MERGED, citations=snapshot,
            knowledge_chunk_id=None, resolution=None, last_seen_at=now,
        )
        await session.flush()
        return RecordResult(root.id, occurrence.id, False,
                            root.duplicate_count, self._status_value(root.status))

    async def _resolve_locked_root(self, repo, row):
        seen: set[int] = set()
        current = row
        for _ in range(_MAX_HOPS):
            if current.id in seen:
                return None
            seen.add(current.id)
            if current.merged_into_id is None:
                return current
            current = await repo.get_for_update(current.merged_into_id)
            if current is None:
                return None
        return None

    async def resolve_root(self, session, row):
        return await self._resolve_locked_root(
            LowConfidenceRepository(session), row
        )

    async def find_message_event(self, session, message_id: int, source: str):
        return (
            await session.execute(
                select(LowConfidenceQuestion).where(
                    LowConfidenceQuestion.message_id == message_id,
                    LowConfidenceQuestion.source == source,
                ).limit(1)
            )
        ).scalars().first()

    async def record_event_isolated(self, **kwargs):
        try:
            async with self._session_factory() as session:
                result = await self.record_event(session, **kwargs)
                await session.commit()
                return result
        except Exception:
            logger.warning("低置信度问题落池失败（不影响本轮回答）", exc_info=True)
            return None

    @staticmethod
    def _status_value(status) -> str:
        return status.value if hasattr(status, "value") else str(status)
