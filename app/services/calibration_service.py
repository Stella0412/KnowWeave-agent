"""置信度阈值校准的后台运行与状态（页面重跑用）。

算法都在 `eval/confidence_calibration.py`；这里只负责线程、进度与状态，
和 `rag_eval_service` 同一套约定：单飞、轮询、异常回传页面。
"""

from __future__ import annotations

import logging
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any

from eval.confidence_calibration import read_report, run

logger = logging.getLogger(__name__)

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "status": "idle",
    "run_id": None,
    "progress": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _report_progress(done: int, total: int) -> None:
    with _state_lock:
        _state["progress"] = {"done": done, "total": total, "message": f"已跑 {done}/{total} 题"}


def _run_guarded(run_id: str, limit: int | None) -> None:
    try:
        run(limit=limit, progress=_report_progress)
        with _state_lock:
            _state.update(status="done", finished_at=_now(), error=None)
    except Exception as exc:  # noqa: BLE001 —— 后台线程必须把错误交回页面
        logger.exception("置信度校准运行失败")
        with _state_lock:
            _state.update(status="error", finished_at=_now(), error=str(exc))


def start(limit: int | None = None) -> tuple[bool, dict]:
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须是正整数")
    with _state_lock:
        if _state["status"] == "running":
            return False, deepcopy(_state)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        _state.update({
            "status": "running",
            "run_id": run_id,
            "progress": {"done": 0, "total": None, "message": "准备中…"},
            "error": None,
            "started_at": _now(),
            "finished_at": None,
        })
        thread = threading.Thread(
            target=_run_guarded, args=(run_id, limit), daemon=True,
            name=f"calibration-{run_id}",
        )
        thread.start()
        return True, deepcopy(_state)


def get_status() -> dict:
    with _state_lock:
        return deepcopy(_state)


def latest() -> dict | None:
    return read_report()
