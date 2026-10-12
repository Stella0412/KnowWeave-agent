"""对话挖知识：历史会话分批喂 LLM 抽问答对 → 暂存表 → 整体去重 → 入 knowledge_chunks。"""

import logging
from dataclasses import dataclass
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.db.models import StagingStatus, VectorizeStatus
from app.db.repository import ChatRepository, KnowledgeRepository
from app.kb.chunking import Chunk
from app.kb.dedup import qa_fingerprint

logger = logging.getLogger(__name__)

MINED_CATEGORY = "对话挖掘"

MINING_SYSTEM_PROMPT = """你是客服知识抽取助手。下面给你一段客服会话记录（USER 是用户说的，ASSISTANT 是客服说的）。

请抽取其中「用户问、客服答」的问答对，规则：
- 只抽对话里真实出现过的内容，不要编造、不要补全、不要改写政策细节或金额
- question 用用户的原话（可去掉「你好」等寒暄）
- answer 用客服的原话，去掉「亲」「您好」等客套，保留全部事实（金额、时效、条件）
- 用户没得到明确答复、只是闲聊、或客服只答「稍等 / 帮您转人工」的，不要抽
- 同一问题反复出现只抽一次
- 一段对话抽不出任何问答对时，返回空列表
"""


class QAPair(BaseModel):
    """从客服对话中抽出的一组问答。"""

    question: str = Field(description="用户在对话中真实提出的问题，保持口语原样")
    answer: str = Field(description="客服给出的答案，必须是对话中真实出现过的说法")


class QABatch(BaseModel):
    """一批对话的抽取结果。没有可抽的问答就返回空列表。"""

    pairs: list[QAPair] = Field(
        default_factory=list, description="本批对话中抽出的问答对；没有就返回空列表"
    )


@dataclass
class MineReport:
    conversations: int = 0
    batches: int = 0
    extracted: int = 0
    kept: int = 0
    discarded: int = 0


class QAMiner:
    def __init__(self, session_factory, structured_model, *, batch_size: int = 20) -> None:
        self._session_factory = session_factory
        self._model = structured_model
        self._batch_size = max(1, batch_size)

    async def mine(self, *, run_id: str | None = None) -> MineReport:
        run_id = run_id or f"mine-{datetime.now():%Y%m%d}"
        report = MineReport()
        transcripts = await self._load_transcripts()
        report.conversations = len(transcripts)

        for index in range(0, len(transcripts), self._batch_size):
            batch_no = f"{run_id}-{index // self._batch_size + 1:03d}"
            batch = transcripts[index : index + self._batch_size]
            report.batches += 1
            try:
                extracted = await self._extract_batch(batch)
            except Exception:
                logger.exception("第 %s 批抽取失败，跳过（可重跑）", batch_no)
                continue
            async with self._session_factory() as session:
                repo = KnowledgeRepository(session)
                await repo.clear_batch(batch_no)
                for source_ref, question, answer in extracted:
                    await repo.add_staging(batch_no, source_ref, question, answer)
                await session.commit()
            report.extracted += len(extracted)

        await self._dedupe(report)
        await self._promote_kept()
        return report

    async def _load_transcripts(self) -> list[tuple[str, str]]:
        """返回 [(source_ref, transcript_text)]，只保留含 user+assistant 的会话。"""
        async with self._session_factory() as session:
            chat = ChatRepository(session)
            conversations = await chat.list_conversations()
            transcripts: list[tuple[str, str]] = []
            for conv in conversations:
                messages = await chat.list_messages(conv.id)
                lines = [
                    f"{m.role.value.upper()}: {m.content}"
                    for m in messages
                    if m.content and m.role.value in ("user", "assistant")
                ]
                roles = {m.role.value for m in messages if m.content}
                if "user" in roles and "assistant" in roles:
                    transcripts.append((f"conversation:{conv.id}", "\n".join(lines)))
        return transcripts

    async def _extract_batch(self, batch: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
        body = "\n\n".join(f"【会话 {ref}】\n{text}" for ref, text in batch)
        messages = [
            SystemMessage(content=MINING_SYSTEM_PROMPT),
            HumanMessage(content=body),
        ]
        result = await self._model.ainvoke(messages)
        batch_result = result if isinstance(result, QABatch) else QABatch.model_validate(result)
        source_ref = batch[0][0] if len(batch) == 1 else f"{batch[0][0]}..{batch[-1][0]}"
        return [
            (source_ref, pair.question.strip(), pair.answer.strip())
            for pair in batch_result.pairs
            if pair.question.strip() and pair.answer.strip()
        ]

    async def _dedupe(self, report: MineReport) -> None:
        """整体去重：与 knowledge_chunks 已有内容 + 暂存表遗留 KEPT 行 + 本轮 staging 逐条比对。

        分两拨，先遗留行后本轮行（先到先得）：
        - **遗留 KEPT 行**：`clear_batch` 只清本 run 的批次号，别的 run_id（每天的 run_id 都不同）
          留下的 KEPT 行会一直躺在暂存表里。它们若已入库 → 标 DISCARDED，否则 `_promote_kept`
          每次重跑都把它们重灌一遍（同一条问答在 KB 里出现 N 次）；若没入库 → 留在 KEPT，
          由本轮补灌（上轮 promote 前崩掉的遗留行因此自动愈合）。
        - **本轮 EXTRACTED 行**：与已入库内容 + 上面留下的行比对。
        """
        async with self._session_factory() as session:
            repo = KnowledgeRepository(session)
            seen = {qa_fingerprint(q, a) for q, a in await repo.list_existing_qa()}

            stale_drop_ids: list[int] = []
            for row in await repo.list_staging(None, StagingStatus.KEPT):
                key = qa_fingerprint(row.question, row.answer)
                if key in seen:
                    stale_drop_ids.append(row.id)
                else:
                    seen.add(key)
            await repo.set_staging_status(stale_drop_ids, StagingStatus.DISCARDED)

            keep_ids: list[int] = []
            drop_ids: list[int] = []
            for row in await repo.list_staging(None, StagingStatus.EXTRACTED):
                key = qa_fingerprint(row.question, row.answer)
                if key in seen:
                    drop_ids.append(row.id)
                else:
                    seen.add(key)
                    keep_ids.append(row.id)
            await repo.set_staging_status(drop_ids, StagingStatus.DISCARDED)
            await repo.set_staging_status(keep_ids, StagingStatus.KEPT)
            await session.commit()
        report.discarded += len(stale_drop_ids) + len(drop_ids)
        report.kept += len(keep_ids)

    async def _promote_kept(self) -> None:
        async with self._session_factory() as session:
            repo = KnowledgeRepository(session)
            rows = await repo.list_staging(None, StagingStatus.KEPT)
            if not rows:
                return
            chunks = [
                Chunk(
                    category=MINED_CATEGORY,
                    questions=row.question,
                    answer=row.answer,
                    section_path=None,
                    content_type="mined",
                    is_key_clause=False,
                )
                for row in rows
            ]
            ids = await repo.insert_chunks(chunks, status=VectorizeStatus.PENDING)
            await repo.link_neighbors(ids)
            await session.commit()
