from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

from app.observability.handler import TurnMetricsHandler
from app.observability.prices import load_prices
from app.services.token_counter import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class ObservabilityBundle:
    handler: TurnMetricsHandler
    callbacks: list
    langfuse_client: object | None
    trace_scope: object


def build_observability(settings, counter=None):
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
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
        os.environ.setdefault("LANGFUSE_HOST", host)
        os.environ.setdefault("LANGFUSE_BASE_URL", host)
        try:
            from langfuse import Langfuse, get_client
            from langfuse.langchain import CallbackHandler

            client = Langfuse(
                public_key=public_key, secret_key=secret_key, host=host,
            )
            langfuse_callback = CallbackHandler()
            trace_id_provider = lambda: get_client().get_current_trace_id()
        except Exception:
            logger.warning("Langfuse 初始化失败，降级为本地指标", exc_info=True)
            client = None
            trace_id_provider = None

    handler = TurnMetricsHandler(
        prices=load_prices(settings.model_prices_path), counter=counter,
        trace_id_provider=trace_id_provider,
    )
    callbacks = [handler]
    if langfuse_callback is not None:
        callbacks.insert(0, langfuse_callback)

    @contextmanager
    def trace_scope(conversation_id):
        if client is None:
            yield
            return
        try:
            from langfuse import propagate_attributes

            cm = propagate_attributes(
                session_id=str(conversation_id),
                trace_name="chat-turn",
                tags=["chat"],
                metadata={"conversation_id": conversation_id},
            )
        except Exception:
            logger.warning("Langfuse trace 注入失败，本轮继续", exc_info=True)
            yield
            return
        try:
            with cm:
                yield
        except Exception:
            logger.warning("Langfuse trace 运行期失败，本轮继续", exc_info=True)
            raise

    return ObservabilityBundle(
        handler=handler,
        callbacks=callbacks,
        langfuse_client=client,
        trace_scope=trace_scope,
    )
