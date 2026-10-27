"""意图标签：意图确定之后创建的 Langfuse 观测打上 intent:<x>。

LangGraph 每个节点在独立 context 里执行，节点内调用 propagate_attributes 不会
传到后续兄弟节点；Langfuse 又只在 span 创建时读取传播属性。因此标签改由
callback handler 直接写：classify_intent 结束拿到 intent 后记到本轮 root run
上，之后每新建一个观测就把标签合并进它的 langfuse.trace.tags。

口径：只统计意图确定之后的开销。指代消解、意图分类两次调用不带标签；
投诉/闲聊路由若没有后续 LLM 调用，成本账里显示为 0。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, get_args
from uuid import UUID

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from langfuse.langchain import CallbackHandler as _LangfuseCallbackHandler
from opentelemetry import trace

from app.prompts.orchestrator import IntentName

logger = logging.getLogger(__name__)

TAG_PREFIX = "intent:"
CLASSIFY_NODE = "classify_intent"
_INTENTS = frozenset(get_args(IntentName))


def extract_intent(outputs: Any) -> str | None:
    """从 classify_intent 的节点输出里取合法意图；拿不准就不打标。"""
    if not isinstance(outputs, Mapping):
        return None
    intent = outputs.get("intent")
    if isinstance(intent, str) and intent in _INTENTS:
        return intent
    return None


def merge_span_tags(span, tag: str) -> bool:
    """把 tag 合并进 span 已有的 langfuse.trace.tags，返回是否真的写入。"""
    existing = getattr(span, "attributes", None) or {}
    tags = [str(item) for item in existing.get(LangfuseOtelSpanAttributes.TRACE_TAGS) or ()]
    if tag in tags:
        return False
    tags.append(tag)
    span.set_attribute(LangfuseOtelSpanAttributes.TRACE_TAGS, tags)
    return True


class IntentTaggingCallbackHandler(_LangfuseCallbackHandler):
    """在 classify_intent 之后新建的观测上补 intent:<意图> 标签。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._intent_tags: dict[UUID, str] = {}
        self._run_nodes: dict[UUID, str] = {}
        self._trace_roots: dict[UUID, Any] = {}

    # -- 新建观测：补标签 -------------------------------------------------
    def _intent_for_run(self, run_id: UUID) -> str | None:
        state = self._get_run_state(run_id)
        if state is None:
            return None
        return self._intent_tags.get(state.root_run_id)

    def _tag_observation(self, run_id: UUID) -> None:
        try:
            tag = self._intent_for_run(run_id)
            if tag is None:
                return
            observation = self._runs.get(run_id)
            span = getattr(observation, "_otel_span", None)
            if span is not None:
                merge_span_tags(span, tag)
        except Exception:
            logger.debug("意图标签写入失败 run=%s", run_id, exc_info=True)

    def _remember_node(
        self,
        run_id: UUID,
        serialized: Mapping | None,
        metadata: Mapping | None,
        kwargs: Mapping[str, Any],
    ) -> None:
        try:
            node = self.get_langchain_run_name(serialized, **kwargs)
            if isinstance(metadata, Mapping):
                meta_node = metadata.get("langgraph_node")
                if isinstance(meta_node, str) and meta_node:
                    node = meta_node
            self._run_nodes[run_id] = node
        except Exception:
            logger.debug("节点名解析失败 run=%s", run_id, exc_info=True)

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        # 根 run 启动时当前 OTel span 就是 handle-chat-turn，先记下来，
        # 等意图确定后把标签补写到它上面（trace 列表据此显示 tags）。
        if parent_run_id is None:
            try:
                span = trace.get_current_span()
                if span is not None and span.is_recording():
                    self._trace_roots[run_id] = span
            except Exception:
                logger.debug("根 span 获取失败 run=%s", run_id, exc_info=True)
        self._remember_node(run_id, serialized, metadata, kwargs)
        super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        self._tag_observation(run_id)

    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        super().on_chat_model_start(
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        self._tag_observation(run_id)

    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        super().on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        self._tag_observation(run_id)

    def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        super().on_tool_start(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        self._tag_observation(run_id)

    def on_retriever_start(
        self,
        serialized,
        query,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        super().on_retriever_start(
            serialized,
            query,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        self._tag_observation(run_id)

    # -- 节点结束：拿到意图 -------------------------------------------------
    def _capture_intent(self, run_id: UUID, outputs: Any) -> None:
        try:
            if self._run_nodes.get(run_id) != CLASSIFY_NODE:
                return
            intent = extract_intent(outputs)
            state = self._get_run_state(run_id)
            if intent is None or state is None:
                return
            tag = f"{TAG_PREFIX}{intent}"
            self._intent_tags[state.root_run_id] = tag
            self._tag_trace_root(state.root_run_id, tag)
        except Exception:
            logger.debug("意图标签捕获失败 run=%s", run_id, exc_info=True)

    def _tag_trace_root(self, root_run_id: UUID, tag: str) -> None:
        """根 span 与 LangGraph 根 run 也补上标签，便于 trace 列表展示。"""
        span = self._trace_roots.get(root_run_id)
        if span is not None:
            try:
                merge_span_tags(span, tag)
            except Exception:
                logger.debug("根 span 意图标签写入失败", exc_info=True)
        observation = self._runs.get(root_run_id)
        otel_span = getattr(observation, "_otel_span", None)
        if otel_span is not None:
            try:
                merge_span_tags(otel_span, tag)
            except Exception:
                logger.debug("LangGraph 根观测意图标签写入失败", exc_info=True)

    def _forget_root(self, root_run_id: UUID) -> None:
        """一轮结束后清掉回调缓存；出错中断也走这里，避免 handler 常驻内存泄漏。"""
        root_state = self._get_root_run_state(root_run_id)
        if root_state is not None:
            for rid in root_state.run_ids:
                self._run_nodes.pop(rid, None)
        self._intent_tags.pop(root_run_id, None)
        self._trace_roots.pop(root_run_id, None)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._capture_intent(run_id, outputs)
        if parent_run_id is None:
            self._forget_root(run_id)
        else:
            self._run_nodes.pop(run_id, None)
        super().on_chain_end(
            outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs,
        )

    def on_chain_error(
        self, error, *, run_id, parent_run_id=None, tags=None, **kwargs,
    ):
        if parent_run_id is None:
            self._forget_root(run_id)
        else:
            self._run_nodes.pop(run_id, None)
        super().on_chain_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs,
        )
