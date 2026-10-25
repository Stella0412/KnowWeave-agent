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
from app.db.models import (
    KnowledgeChunk,
    LowConfidenceQuestion,
    LowConfidenceStatus,
    StagingStatus,
    VectorizeStatus,
)
from app.db.repository import (
    FeedbackRepository,
    KnowledgeRepository,
    LowConfidenceRepository,
)
from app.kb.chunking import Chunk
from app.kb.dedup import normalize_text
from app.kb.dualwrite import vectorize_pending as vectorize_pending_rows
from app.kb.milvus_client import get_milvus_client
from app.rag.evidence import cap_citations

logger = logging.getLogger(__name__)
_MAX_HOPS = 10


def question_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def qa_fingerprint(question: str, answer: str) -> str:
    return question_fingerprint(f"{question}\n{answer}")


class ReviewRejected(Exception):
    def __init__(self, status_code: int, detail: str, extra: dict | None = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.extra = extra or {}


class ReviewOrphan(ReviewRejected):
    pass


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
        self._vectorize_lock = asyncio.Lock()
        self._cluster_locks: dict[int, asyncio.Lock] = {}

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
                  "skipped_cross_status": 0, "skipped_locked": 0,
                  "degraded": False, "_cluster_ids": set()}
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
                        # 固定加锁顺序：组内根按 id 升序（spec §6.1），避免两个归并事务互相死锁。
                        group = sorted(group, key=lambda r: int(r.id))
                        canonical = _canonical(group)
                        losers = [r for r in group if r.id != canonical.id]
                        report["_cluster_ids"].add(canonical.id)
                        if dry_run:
                            report["semantic_merged"] += len(losers)
                            report["clusters"] += 1
                            continue
                        merged = False
                        for attempt in range(2):
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
                                merged = True
                                break
                            except (OperationalError, DBAPIError):
                                logger.warning(
                                    "语义归并组被锁（第 %s 次尝试）ids=%s",
                                    attempt + 1, [r.id for r in group], exc_info=True,
                                )
                                if attempt == 0:
                                    for row in group:
                                        try:
                                            await session.refresh(row)
                                        except Exception:
                                            pass
                        if not merged:
                            report["skipped_locked"] += 1
                            continue
                        report["semantic_merged"] += len(losers)
                        report["clusters"] += 1
            if not dry_run:
                await session.commit()
        report["duration_ms"] = int((time.perf_counter() - start) * 1000)
        report["_cluster_ids"] = sorted(report["_cluster_ids"])
        return report

    def _cluster_lock(self, cluster_id: int) -> asyncio.Lock:
        lock = self._cluster_locks.get(cluster_id)
        if lock is None:
            lock = asyncio.Lock()
            self._cluster_locks[cluster_id] = lock
        return lock

    async def _resolve_cluster(self, session, cluster_id: int, *, for_update: bool = False):
        repo = LowConfidenceRepository(session)
        row = (
            await repo.get_for_update(cluster_id)
            if for_update
            else await repo.get(cluster_id)
        )
        if row is None:
            raise ReviewRejected(404, "cluster 不存在")
        if row.merged_into_id is None:
            return row
        root = await self.resolve_root(session, row)
        if root is None:
            if for_update:
                # 详情页可以把孤儿行指出来（404 + orphan_id）；写操作没有根就没法落状态机。
                raise ReviewRejected(409, "cluster root not found")
            raise ReviewOrphan(
                404, "cluster root not found", extra={"orphan_id": cluster_id}
            )
        return root

    async def list_queue(self, *, status="待处理", limit=50, offset=0):
        async with self._session_factory() as session:
            repo = LowConfidenceRepository(session)
            roots = await repo.list_roots(status=status, limit=limit, offset=offset)
            total = await repo.count_roots(status=status)
            items = []
            for root in roots:
                children = await repo.list_children(root.id)
                rows = [root, *children]
                sources: dict[str, int] = {}
                for row in rows:
                    key = row.source.value if hasattr(row.source, "value") else str(row.source)
                    sources[key] = sources.get(key, 0) + 1
                snapshots = [r for r in rows if r.citations]
                latest = max(snapshots, key=lambda r: r.last_seen_at or datetime.min, default=None)
                items.append({
                    "id": root.id,
                    "raw_question": root.raw_question,
                    "duplicate_count": int(root.duplicate_count or 0),
                    "status": self._status_value(root.status),
                    "sources": sources,
                    "conversation_ids": sorted({r.conversation_id for r in rows if r.conversation_id}),
                    "latest_citations": list(latest.citations) if latest is not None else None,
                    "citations_count": len(latest.citations) if latest is not None else 0,
                    "created_at": root.created_at.isoformat() if root.created_at else None,
                    "last_seen_at": root.last_seen_at.isoformat() if root.last_seen_at else None,
                })
        return items, int(total)

    async def cluster_detail(self, cluster_id: int) -> dict:
        async with self._session_factory() as session:
            root = await self._resolve_cluster(session, cluster_id)
            repo = LowConfidenceRepository(session)
            children = await repo.list_children(root.id)
            rows = [root, *children]
            variants = list(dict.fromkeys(
                (r.raw_question or "").strip() for r in rows if (r.raw_question or "").strip()
            ))
            states = [
                "has_snippets" if r.citations else ("missing" if r.citations is None else "empty")
                for r in rows
            ]
            if "has_snippets" in states:
                evidence_state = "has_snippets"
            elif "empty" in states:
                evidence_state = "empty"
            else:
                evidence_state = "missing"
            message_ids = [r.message_id for r in rows if r.message_id is not None]
            comments = await FeedbackRepository(session).comments_for_messages(message_ids)
            chunk = (
                await session.get(KnowledgeChunk, root.knowledge_chunk_id)
                if root.knowledge_chunk_id is not None else None
            )
            return {
                "id": root.id,
                "redirected_from": cluster_id if cluster_id != root.id else None,
                "raw_question": root.raw_question,
                "status": self._status_value(root.status),
                "duplicate_count": int(root.duplicate_count or 0),
                "resolution": root.resolution,
                "variants": variants,
                "evidence_state": evidence_state,
                "occurrences": [{
                    "id": r.id,
                    "conversation_id": r.conversation_id,
                    "message_id": r.message_id,
                    "source": r.source.value if hasattr(r.source, "value") else str(r.source),
                    "reason": r.reason,
                    "citations": list(r.citations) if r.citations else r.citations,
                    "comment": comments.get(r.message_id),
                    "last_seen_at": r.last_seen_at.isoformat() if r.last_seen_at else None,
                } for r in rows],
                "knowledge_chunk": None if chunk is None else {
                    "id": chunk.id,
                    "question": chunk.questions,
                    "answer": chunk.answer,
                    "vectorize_status": (
                        chunk.vectorize_status.value
                        if hasattr(chunk.vectorize_status, "value") else str(chunk.vectorize_status)
                    ),
                    "vector_id": chunk.vector_id,
                },
            }

    async def metrics(self) -> dict:
        async with self._session_factory() as session:
            repo = LowConfidenceRepository(session)
            roots = await repo.list_roots(status=None, limit=10000, offset=0)
            total_signals = sum(int(r.duplicate_count or 0) for r in roots)
            status_counts: dict[str, int] = {}
            source_counts: dict[str, int] = {}
            for root in roots:
                status = self._status_value(root.status)
                status_counts[status] = status_counts.get(status, 0) + 1
                rows = [root, *(await repo.list_children(root.id))]
                for row in rows:
                    key = row.source.value if hasattr(row.source, "value") else str(row.source)
                    source_counts[key] = source_counts.get(key, 0) + 1
        return {
            "total_signals": total_signals,
            "root_total": len(roots),
            "status_counts": status_counts,
            "source_counts": source_counts,
        }

    async def submit_answer(self, cluster_id: int, *, question, answer, category="人工补充") -> dict:
        async with self._cluster_lock(cluster_id):
            async with self._session_factory() as session:
                root = await self._resolve_cluster(session, cluster_id, for_update=True)
                if root.status is not LowConfidenceStatus.PENDING:
                    raise ReviewRejected(409, "只有待处理问题可以补答")
                q = (question or "").strip() or (root.raw_question or "").strip()
                a = (answer or "").strip()
                if not q or not a:
                    raise ReviewRejected(422, "问题与答案不能为空")
                signature = qa_fingerprint(q, a)
                existing = await KnowledgeRepository(session).list_existing_qa_with_ids()
                reused = next(((cid, status) for cid, qs, ans, status in existing
                               if qa_fingerprint(qs, ans) == signature), None)
                if reused:
                    chunk_id, reused_status = reused
                    root.status = LowConfidenceStatus.CLOSED
                    root.knowledge_chunk_id = chunk_id
                    root.resolution = f"复用已有知识，chunk #{chunk_id}"
                    await session.commit()
                    reused_flag = True
                else:
                    batch = "flywheel-" + datetime.now().strftime("%Y%m%d")
                    await KnowledgeRepository(session).add_staging(
                        batch, f"low_conf:{cluster_id}", q, a, status=StagingStatus.KEPT
                    )
                    chunk_id = (await KnowledgeRepository(session).insert_chunks([
                        Chunk(category=category or "人工补充", questions=q, answer=a,
                              content_type="gap_review",
                              section_path=f"人工补充 > 飞轮#{cluster_id}",
                              is_key_clause=False)
                    ], status=VectorizeStatus.PENDING))[0]
                    root.status = LowConfidenceStatus.CLOSED
                    root.knowledge_chunk_id = chunk_id
                    root.resolution = f"人工补充入库，chunk #{chunk_id}"
                    await session.commit()
                    reused_flag = False
                    reused_status = None
        if reused_status is VectorizeStatus.DONE:
            vectorized = True
        else:
            vectorized = await self._try_vectorize([chunk_id])
        vector_status = "done" if vectorized else "pending"
        return {
            "status": "已关闭", "knowledge_chunk_id": chunk_id,
            "reused": reused_flag, "vectorized": vectorized,
            "vectorize_status": vector_status,
        }

    async def _try_vectorize(self, ids: list[int]) -> bool:
        if self._vectorize_lock.locked():
            return False
        try:
            async with self._vectorize_lock:
                await vectorize_pending_rows(
                    self._session_factory,
                    embed=self._embed,
                    milvus=self._milvus_factory(),
                    ids=ids,
                )
            return True
        except Exception:
            logger.warning("补答向量化失败，chunk 保持 pending", exc_info=True)
            return False

    async def vectorize_pending(self) -> dict:
        if self._vectorize_lock.locked():
            raise ReviewRejected(409, "已有向量化在运行")
        try:
            async with self._vectorize_lock:
                count = await vectorize_pending_rows(
                    self._session_factory,
                    embed=self._embed,
                    milvus=self._milvus_factory(),
                )
            return {"vectorized": count}
        except Exception as exc:
            raise ReviewRejected(503, f"向量化失败：{exc}") from exc

    async def ignore(self, cluster_id: int, resolution: str) -> dict:
        text = (resolution or "").strip()
        if not 1 <= len(text) <= 500:
            raise ReviewRejected(422, "处置说明长度必须为 1–500 字")
        async with self._cluster_lock(cluster_id):
            async with self._session_factory() as session:
                root = await self._resolve_cluster(session, cluster_id, for_update=True)
                if root.status is not LowConfidenceStatus.PENDING:
                    raise ReviewRejected(409, "只有待处理问题可以驳回")
                root.status = LowConfidenceStatus.IGNORED
                root.resolution = text
                await session.commit()
                return {"id": root.id, "status": self._status_value(root.status)}

    async def reopen(self, cluster_id: int, resolution: str) -> dict:
        text = (resolution or "").strip()
        if not 1 <= len(text) <= 500:
            raise ReviewRejected(422, "处置说明长度必须为 1–500 字")
        async with self._cluster_lock(cluster_id):
            async with self._session_factory() as session:
                root = await self._resolve_cluster(session, cluster_id, for_update=True)
                if root.status is LowConfidenceStatus.PENDING:
                    raise ReviewRejected(409, "问题已经是待处理")
                root.status = LowConfidenceStatus.PENDING
                root.resolution = text
                await session.commit()
                return {"id": root.id, "status": self._status_value(root.status)}
