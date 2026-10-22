import json

import pytest

from app.config import Settings
from app.core.logging import configure_context_logging, log_event
from app.services.context_manager import log_startup_budget, measure_system_prefix
from app.services.token_counter import TokenCounter


def test_file_handler_writes_json_lines(tmp_path):
    path = tmp_path / "app.log"
    logger = configure_context_logging(str(path))
    log_event(logger, "history_ctx", conversation_id=1, summary="订单1001")
    for handler in logger.handlers:
        handler.flush()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    assert payload["event"] == "history_ctx"
    assert payload["summary"] == "订单1001"


def test_measure_system_prefix_is_positive():
    counter = TokenCounter(estimate_factor=1.0)
    assert measure_system_prefix(counter, session_factory=None) > 500


def test_startup_budget_refuses_when_prefix_exceeds_reserve(tmp_path):
    logger = configure_context_logging(str(tmp_path / "app.log"))
    s = Settings(_env_file=None, system_prefix_reserve_tokens=1)
    counter = TokenCounter(estimate_factor=1.0)
    with pytest.raises(RuntimeError, match="系统前缀超出预算"):
        log_startup_budget(logger, s, counter)
