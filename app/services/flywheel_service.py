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
from sqlalchemy.exc import DBAPIError, OperationalError

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


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _canonical(rows):
    return sorted(rows, key=lambda r: (
        0 if r.knowledge_chunk_id is not None else 1,
        -int(r.duplicate_count or 0),
        int(r.id),
    ))[0]


def _resolve_in_memory(row, by_id):
    seen: set[int] = set()
    current = row
    for _ in range(_MAX_HOPS):
        if current.id in seen:
            return None
        seen.add(current.id)
        if current.merged_into_id is None:
            return current
        current = by_id.get(current.merged_into_id)
        if current is None:
            return None
    return None


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

    async def backfill_fingerprints(self) -> int:
        async with self._session_factory() as session:
            rows = (await session.execute(select(LowConfidenceQuestion))).scalars().all()
            changed = 0
            for row in rows:
                fp = question_fingerprint(row.raw_question or "")
                if row.fingerprint != fp:
                    row.fingerprint = fp
                    changed += 1
            await session.commit()
            return changed

    async def merge_exact(self, *, dry_run: bool = False) -> dict:
        start = time.perf_counter()
        report = {
            "exact_merged": 0, "clusters": 0, "skipped": 0,
            "skipped_cross_status": 0, "skipped_locked": 0,
            "_cluster_ids": set(),
        }
        async with self._session_factory() as session:
            rows = (await session.execute(
                select(LowConfidenceQuestion)
                .where(LowConfidenceQuestion.merged_into_id.is_(None))
                .order_by(LowConfidenceQuestion.id)
                .with_for_update()
            )).scalars().all()
            by_fp: dict[str, list] = {}
            for row in rows:
                fp = row.fingerprint or question_fingerprint(row.raw_question or "")
                by_fp.setdefault(fp, []).append(row)
            # 同一指纹下不同状态之间不归并，但要把“本可归并”的组对数记进报告。
            for members in by_fp.values():
                counts: dict[LowConfidenceStatus, int] = {}
                for row in members:
                    counts[row.status] = counts.get(row.status, 0) + 1
                status_counts = list(counts.values())
                for i in range(len(status_counts)):
                    for j in range(i + 1, len(status_counts)):
                        report["skipped_cross_status"] += status_counts[i] * status_counts[j]
            repo = LowConfidenceRepository(session)
            for fingerprint, members in by_fp.items():
                by_status: dict[LowConfidenceStatus, list] = {}
                for row in members:
                    by_status.setdefault(row.status, []).append(row)
                for group in by_status.values():
                    if len(group) < 2:
                        continue
                    canonical = _canonical(group)
                    losers = [r for r in group if r.id != canonical.id]
                    report["_cluster_ids"].add(canonical.id)
                    if dry_run:
                        report["exact_merged"] += len(losers)
                        report["clusters"] += 1
                        continue
                    try:
                        async with session.begin_nested():
                            canonical.duplicate_count += sum(
                                int(l.duplicate_count or 0) for l in losers
                            )
                            canonical.last_seen_at = max(
                                [canonical.last_seen_at] + [l.last_seen_at for l in losers]
                            )
                            for loser in losers:
                                await repo.repoint_children(loser.id, canonical.id)
                                loser.status = LowConfidenceStatus.MERGED
                                loser.merged_into_id = canonical.id
                                loser.duplicate_count = 1
                                if (
                                    canonical.knowledge_chunk_id is None
                                    and loser.knowledge_chunk_id is not None
                                ):
                                    canonical.knowledge_chunk_id = loser.knowledge_chunk_id
                                    canonical.resolution = loser.resolution
                    except (OperationalError, DBAPIError):
                        logger.warning("精确归并组被锁跳过 fp=%s", fingerprint, exc_info=True)
                        report["skipped_locked"] += 1
                        continue
                    report["exact_merged"] += len(losers)
                    report["clusters"] += 1
            if not dry_run:
                await session.commit()
        report["duration_ms"] = int((time.perf_counter() - start) * 1000)
        report["_cluster_ids"] = sorted(report["_cluster_ids"])
        return report

    async def merge_semantic(self, *, threshold, limit, batch_size, dry_run=False):
        start = time.perf_counter()
        report = {"semantic_merged": 0, "clusters": 0, "skipped": 0,
                  "skipped_cross_status": 0, "degraded": False,
                  "_cluster_ids": set()}
        async with self._session_factory() as session:
            rows = (await session.execute(
                select(LowConfidenceQuestion)
                .order_by(LowConfidenceQuestion.last_seen_at.desc(),
                          LowConfidenceQuestion.id.desc())
                .limit(limit)
                .with_for_update()
            )).scalars().all()
            by_id = {row.id: row for row in rows}
            valid = [r for r in rows if (r.raw_question or "").strip()]
            report["skipped"] = len(rows) - len(valid)
            vectors: list[list[float]] = []
            try:
                for i in range(0, len(valid), batch_size):
                    batch = [r.raw_question for r in valid[i:i + batch_size]]
                    vectors.extend(await asyncio.to_thread(self._embed, batch))
            except Exception:
                logger.warning("语义归并 embedding 降级，精确结果保留", exc_info=True)
                report["degraded"] = True
                vectors = []
            if vectors:
                parent = list(range(len(valid)))

                def find(x: int) -> int:
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                def union(a: int, b: int) -> None:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra

                for i in range(len(valid)):
                    for j in range(i + 1, len(valid)):
                        if _cosine(vectors[i], vectors[j]) >= threshold:
                            union(i, j)
                components: dict[int, list] = {}
                for index, row in enumerate(valid):
                    components.setdefault(find(index), []).append(row)
                repo = LowConfidenceRepository(session)
                for members in components.values():
                    if len(members) < 2:
                        continue
                    roots: dict[int, LowConfidenceQuestion] = {}
                    for row in members:
                        root = _resolve_in_memory(row, by_id)
                        if root is None:
                            report["skipped"] += 1
                            continue
                        roots[root.id] = root
                    by_status: dict[LowConfidenceStatus, list] = {}
                    for root in roots.values():
                        by_status.setdefault(root.status, []).append(root)
                    status_groups = list(by_status.values())
                    for i in range(len(status_groups)):
                        for j in range(i + 1, len(status_groups)):
                            report["skipped_cross_status"] += (
                                len(status_groups[i]) * len(status_groups[j])
                            )
                    for group in status_groups:
                        if len(group) < 2:
                            continue
                        canonical = _canonical(group)
                        losers = [r for r in group if r.id != canonical.id]
                        report["_cluster_ids"].add(canonical.id)
                        if dry_run:
                            report["semantic_merged"] += len(losers)
                            report["clusters"] += 1
                            continue
                        canonical.duplicate_count += sum(
                            int(l.duplicate_count or 0) for l in losers
                        )
                        canonical.last_seen_at = max(
                            [canonical.last_seen_at] + [l.last_seen_at for l in losers]
                        )
                        for loser in losers:
                            await repo.repoint_children(loser.id, canonical.id)
                            loser.status = LowConfidenceStatus.MERGED
                            loser.merged_into_id = canonical.id
                            loser.duplicate_count = 1
                            if (
                                canonical.knowledge_chunk_id is None
                                and loser.knowledge_chunk_id is not None
                            ):
                                canonical.knowledge_chunk_id = loser.knowledge_chunk_id
                                canonical.resolution = loser.resolution
                        report["semantic_merged"] += len(losers)
                        report["clusters"] += 1
            if not dry_run:
                await session.commit()
        report["duration_ms"] = int((time.perf_counter() - start) * 1000)
        report["_cluster_ids"] = sorted(report["_cluster_ids"])
        return report
