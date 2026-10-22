"""ch07 上下文日志：控制台 + log/app.log 轮转文件，单行 JSON。"""

from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 3


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def configure_context_logging(log_file: str, *, logger_name: str = "knowweave.context"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    target = _norm(log_file)
    has_file = any(
        isinstance(handler, RotatingFileHandler)
        and _norm(getattr(handler, "baseFilename", "") or "") == target
        for handler in logger.handlers
    )
    if not has_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    if not any(
        type(handler) is logging.StreamHandler for handler in logger.handlers
    ):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream)
    return logger


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False))
