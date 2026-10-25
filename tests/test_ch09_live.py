# tests/test_ch09_live.py
"""ch09 真实模型闭环测试（默认 skip，需 `CH09_LIVE=1` 与真实模型/embedding）。

前置：先停应用（Milvus Lite 单进程独占锁），MySQL 与 Langfuse 已启动，
`.env` 中存在 LLM/embedding key。运行：

    CH09_LIVE=1 uv run pytest -q tests/test_ch09_live.py

闭环路径：跑一条注定答不出的问题 → 拒答 → 以 self_check 来源入池建根 →
人工补答并向量化 → 再问同一问题 → 回答且引用新 chunk → 清理测试 chunk。
"""
import asyncio
import os
import uuid

import pytest

from app.config import settings
from app.db.engine import SessionFactory
from app.kb.milvus_client import get_milvus_client
from app.rag.pipeline import run_pipeline
from app.services.flywheel_service import FlywheelService


pytestmark = pytest.mark.skipif(
    os.getenv("CH09_LIVE") != "1", reason="需要 CH09_LIVE=1 与真实模型/embedding"
)


async def test_unanswerable_question_becomes_answerable():
    assert settings.llm_api_key and settings.embed_api_key  # 只验非空，不打印
    question = f"第九章闭环测试-{uuid.uuid4().hex[:8]} 的保价规则是什么"
    before = await asyncio.to_thread(run_pipeline, question)
    assert before.refused

    service = FlywheelService(SessionFactory, settings=settings)
    event = await service.record_event_isolated(
        conversation_id=None, raw_question=question, source="self_check",
        reason="ch09 闭环验收", citations=before.citations,
    )
    assert event is not None and event.created is True
    result = await service.submit_answer(
        event.root_id,
        question=question,
        answer="该测试问题的标准答案：下单 7 天内可申请价保。",
        category="人工补充",
    )
    assert result["vectorized"] is True
    chunk_id = result["knowledge_chunk_id"]
    after = await asyncio.to_thread(run_pipeline, question)
    assert after.refused is False
    assert any(c["chunk_id"] == chunk_id for c in after.citations)
    try:
        get_milvus_client().delete_ids([chunk_id])
    except Exception:
        pass
