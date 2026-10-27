"""校准后台线程：单飞、进度、失败回传页面。"""

import threading
import time

import pytest

from app.services import calibration_service as svc


@pytest.fixture(autouse=True)
def _reset_state():
    def _idle():
        with svc._state_lock:
            svc._state.update({
                "status": "idle", "run_id": None, "progress": None,
                "error": None, "started_at": None, "finished_at": None,
            })

    _idle()
    yield
    _idle()


def _wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_start_runs_in_background_and_reports_done(monkeypatch):
    calls = []

    def fake_run(*, limit=None, progress=None):
        calls.append(limit)
        progress(1, 2)
        return {}

    monkeypatch.setattr(svc, "run", fake_run)
    accepted, state = svc.start(limit=5)
    assert accepted is True
    assert state["status"] == "running"
    assert _wait_until(lambda: svc.get_status()["status"] == "done")
    assert calls == [5]
    assert svc.get_status()["error"] is None


def test_second_start_is_rejected_while_running(monkeypatch):
    gate = threading.Event()

    def fake_run(*, limit=None, progress=None):
        gate.wait(2)
        return {}

    monkeypatch.setattr(svc, "run", fake_run)
    accepted, _ = svc.start()
    accepted_again, state = svc.start()
    assert accepted is True
    assert accepted_again is False
    assert state["status"] == "running"
    gate.set()
    assert _wait_until(lambda: svc.get_status()["status"] == "done")


def test_failure_is_reported_to_the_page(monkeypatch):
    def boom(*, limit=None, progress=None):
        raise RuntimeError("模型挂了")

    monkeypatch.setattr(svc, "run", boom)
    svc.start()
    assert _wait_until(lambda: svc.get_status()["status"] == "error")
    assert "模型挂了" in svc.get_status()["error"]


def test_non_positive_limit_is_rejected():
    with pytest.raises(ValueError):
        svc.start(limit=0)


def test_latest_reads_the_artifact(monkeypatch):
    monkeypatch.setattr(svc, "read_report", lambda: {"recommended": 0.42})
    assert svc.latest() == {"recommended": 0.42}
