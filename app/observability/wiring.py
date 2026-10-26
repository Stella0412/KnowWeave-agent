from __future__ import annotations

import importlib.metadata
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

from app.observability.handler import TurnMetricsHandler
from app.observability.masking import mask_otel_spans
from app.observability.prices import load_prices
from app.services.token_counter import TokenCounter

logger = logging.getLogger(__name__)

_client: object | None = None


def get_langfuse_client():
    """供 RAG / 工具等非 LangChain 代码复用同一个 Langfuse 客户端。"""
    return _client


def _release(settings) -> str | None:
    if settings.app_release:
        return settings.app_release
    try:
        return importlib.metadata.version("knowweave")
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass
class ObservabilityBundle:
    handler: TurnMetricsHandler
    callbacks: list
    langfuse_client: object | None
    trace_scope: object


def build_observability(settings, counter=None):
    global _client
    _client = None
    counter = counter or TokenCounter(
        cjk_tokens_per_char=settings.cjk_tokens_per_char,
        estimate_factor=settings.token_estimate_factor,
    )
    client = None
    trace_id_provider = None
    langfuse_callback = None
    if (
        settings.langfuse_tracing_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    ):
        public_key = settings.langfuse_public_key
        secret_key = settings.langfuse_secret_key
        host = settings.langfuse_base_url
        environment = (settings.app_env or "").strip() or "development"
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
        os.environ.setdefault("LANGFUSE_HOST", host)
        os.environ.setdefault("LANGFUSE_BASE_URL", host)
        try:
            from langfuse import Langfuse, get_client
            from langfuse.langchain import CallbackHandler

            client = Langfuse(
                public_key=public_key, secret_key=secret_key, host=host,
                environment=environment, release=_release(settings),
                mask_otel_spans=mask_otel_spans,
            )
            _client = client
            langfuse_callback = CallbackHandler()
            trace_id_provider = lambda: get_client().get_current_trace_id()
        except Exception:
            logger.warning("Langfuse 初始化失败，降级为本地指标", exc_info=True)
            client = None
            _client = None
            trace_id_provider = None

    handler = TurnMetricsHandler(
        prices=load_prices(settings.model_prices_path), counter=counter,
        trace_id_provider=trace_id_provider,
    )
    callbacks = [handler]
    if langfuse_callback is not None:
        callbacks.insert(0, langfuse_callback)

    @contextmanager
    def trace_scope(conversation_id, message, *, user_id=None, mode="chat"):
        if client is None:
            yield None
            return
        try:
            from langfuse import propagate_attributes
        except Exception:
            logger.warning("Langfuse trace 注入失败，本轮继续", exc_info=True)
            yield None
            return

        try:
            root_cm = client.start_as_current_observation(
                as_type="span",
                name="handle-chat-turn",
                input=[{"role": "user", "content": message}],
                metadata={"conversation_id": conversation_id},
            )
            root = root_cm.__enter__()
        except Exception:
            logger.warning("Langfuse trace 创建失败，本轮继续", exc_info=True)
            yield None
            return

        attrs_cm = None
        try:
            attrs_cm = propagate_attributes(
                session_id=str(conversation_id),
                user_id=str(user_id) if user_id else None,
                trace_name="handle-chat-turn",
                tags=["chat", f"mode:{mode}"],
                metadata={"conversation_id": conversation_id},
            )
            attrs_cm.__enter__()
        except Exception:
            logger.warning("Langfuse trace 属性注入失败，本轮继续", exc_info=True)
            if attrs_cm is not None:
                try:
                    attrs_cm.__exit__(None, None, None)
                except Exception:
                    pass
            try:
                root_cm.__exit__(None, None, None)
            except Exception:
                pass
            yield None
            return

        exit_args = (None, None, None)
        try:
            yield root
        except Exception as exc:
            try:
                root.update(
                    level="ERROR",
                    status_message=type(exc).__name__[:500],
                )
            except Exception:
                logger.debug("trace 错误标记失败", exc_info=True)
            exit_args = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            try:
                attrs_cm.__exit__(*exit_args)
            except Exception:
                logger.debug("trace 属性清理失败", exc_info=True)
            try:
                root_cm.__exit__(*exit_args)
            except Exception:
                logger.debug("trace 结束失败", exc_info=True)

    return ObservabilityBundle(
        handler=handler,
        callbacks=callbacks,
        langfuse_client=client,
        trace_scope=trace_scope,
    )
