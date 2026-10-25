import json
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)
_MAX = 500


def _cut(value):
    return value if len(value) <= _MAX else value[:_MAX]


def _summarize_args(args: dict) -> dict:
    text = json.dumps(args, ensure_ascii=False, default=str, sort_keys=True)
    return {"args": _cut(text)}


@contextmanager
def tool_span(client, name: str, *, input: dict, metadata: dict):
    if client is None:
        yield None
        return
    span = None
    try:
        span = client.start_span(name=name, input=input, metadata=metadata)
    except Exception:
        logger.debug("Langfuse 工具 span 创建失败", exc_info=True)
        yield None
        return
    try:
        yield span
    except Exception as exc:
        try:
            span.update(level="ERROR", status_message=_cut(type(exc).__name__))
        except Exception:
            logger.debug("工具 span 错误更新失败", exc_info=True)
        raise
    finally:
        try:
            span.end()
        except Exception:
            logger.debug("工具 span end 失败", exc_info=True)
