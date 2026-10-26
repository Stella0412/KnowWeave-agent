import logging
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)
_MAX = 500
_current_observation: ContextVar[object | None] = ContextVar(
    "knowweave_current_observation", default=None,
)


def _cut(value):
    return value if len(value) <= _MAX else value[:_MAX]


def _client():
    """延迟导入，避免 spans ↔ wiring 的循环依赖。"""
    try:
        from app.observability.wiring import get_langfuse_client

        return get_langfuse_client()
    except Exception:
        return None


def _run_parent(client):
    """取当前 LangChain run 对应的 Langfuse observation。

    LangChain 的 Langfuse handler 用 run_id → observation 的映射建父子关系，并不会把
    span 挂进 OTel 的当前上下文；所以 LangGraph 节点里的手写 span 如果只看 OTel
    上下文，会直接挂到 trace 根上。这里顺着 LangChain 的 config contextvar 找到当前
    节点的 observation，让 RAG/工具 span 落在正确的节点下面。
    """
    try:
        from langchain_core.runnables.config import var_child_runnable_config

        config = var_child_runnable_config.get()
    except Exception:
        return None
    if not config:
        return None
    callbacks = config.get("callbacks")
    run_id = getattr(callbacks, "parent_run_id", None)
    if run_id is None:
        return None
    for handler in getattr(callbacks, "handlers", None) or ():
        lookup = getattr(handler, "_get_parent_observation", None)
        if lookup is None:
            continue
        try:
            observation = lookup(run_id)
        except Exception:
            continue
        if observation is not None and observation is not client:
            return observation
    return None


def _active_parent(client):
    """优先复用本模块刚开出来的 observation，让嵌套 span 落在最近一层。"""
    observation = _current_observation.get()
    if observation is not None:
        return observation
    return _run_parent(client)


def _summarize_args(args: dict) -> dict:
    """工具参数保留结构（便于在 UI 里读），只截断超长字符串。"""

    def clean(value):
        if isinstance(value, str):
            return _cut(value)
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(args or {})


@contextmanager
def tool_span(client, name: str, *, input: dict, metadata: dict):
    if client is None:
        yield None
        return
    span = None
    try:
        source = _active_parent(client) or client
        span = source.start_observation(
            as_type="tool", name=name, input=input, metadata=metadata,
        )
    except Exception:
        logger.debug("Langfuse 工具 span 创建失败", exc_info=True)
        yield None
        return
    token = _current_observation.set(span)
    try:
        yield span
    except Exception as exc:
        try:
            span.update(level="ERROR", status_message=_cut(type(exc).__name__))
        except Exception:
            logger.debug("工具 span 错误更新失败", exc_info=True)
        raise
    finally:
        _current_observation.reset(token)
        try:
            span.end()
        except Exception:
            logger.debug("工具 span end 失败", exc_info=True)


@contextmanager
def observation_scope(
    as_type: str,
    name: str,
    *,
    input=None,
    metadata=None,
    model: str | None = None,
):
    """在当前 OTel 上下文里开一个带类型的子 observation；未配置 Langfuse 时为空操作。"""
    client = _client()
    if client is None:
        yield None
        return
    try:
        source = _active_parent(client) or client
        cm = source.start_as_current_observation(
            as_type=as_type, name=name, input=input, metadata=metadata, model=model,
        )
        observation = cm.__enter__()
    except Exception:
        logger.debug("Langfuse observation 创建失败", exc_info=True)
        yield None
        return

    token = _current_observation.set(observation)
    exit_args = (None, None, None)
    try:
        yield observation
    except Exception as exc:
        try:
            observation.update(level="ERROR", status_message=_cut(type(exc).__name__))
        except Exception:
            logger.debug("observation 错误更新失败", exc_info=True)
        exit_args = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        _current_observation.reset(token)
        try:
            cm.__exit__(*exit_args)
        except Exception:
            logger.debug("Langfuse observation end 失败", exc_info=True)
