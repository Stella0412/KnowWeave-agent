"""评估面板的 HTTP 接口：页面、最近一次结果、运行状态、触发重跑。

不启动真正的评估线程——线程里的 Milvus/LLM 依赖在 service 的测试之外，
这里只钉住路由契约：什么参数、什么状态码、并发重跑怎么拒。
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import rag_eval


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(rag_eval.router)
    return TestClient(app)


def test_page_is_served():
    resp = _client().get("/rag-eval")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "RAG 评估" in resp.text


def test_latest_returns_null_before_first_run(monkeypatch):
    monkeypatch.setattr(rag_eval.service, "read_latest", lambda: None)
    resp = _client().get("/rag-eval/api/latest")
    assert resp.status_code == 200
    assert resp.json() == {"latest": None}


def test_latest_returns_saved_result(monkeypatch):
    monkeypatch.setattr(rag_eval.service, "read_latest", lambda: {"run_id": "r1", "samples": 39})
    resp = _client().get("/rag-eval/api/latest")
    assert resp.json()["latest"]["run_id"] == "r1"


def test_status_reports_idle(monkeypatch):
    monkeypatch.setattr(
        rag_eval.service,
        "get_status",
        lambda: {"status": "idle", "progress": None},
    )
    resp = _client().get("/rag-eval/api/status")
    assert resp.json()["status"] == "idle"


def test_run_rejects_unknown_mode(monkeypatch):
    called = []
    monkeypatch.setattr(rag_eval.service, "start_run", lambda mode: called.append(mode) or (True, {}))
    resp = _client().post("/rag-eval/api/run", json={"mode": "everything"})
    assert resp.status_code == 400
    assert "everything" in resp.json()["detail"]  # 错误信息带上被拒的值
    assert called == []


def test_run_conflicts_when_already_running(monkeypatch):
    monkeypatch.setattr(
        rag_eval.service,
        "start_run",
        lambda mode: (False, {"status": "running", "mode": mode}),
    )
    resp = _client().post("/rag-eval/api/run", json={"mode": "retrieval"})
    assert resp.status_code == 409
    assert resp.json()["detail"]


def test_run_accepts_and_returns_run_state(monkeypatch):
    monkeypatch.setattr(
        rag_eval.service,
        "start_run",
        lambda mode: (True, {"status": "running", "mode": mode, "run_id": "r2"}),
    )
    resp = _client().post("/rag-eval/api/run", json={"mode": "full"})
    assert resp.status_code == 202
    assert resp.json()["run_id"] == "r2"
    assert resp.json()["mode"] == "full"


def test_page_without_trailing_slash_api_is_not_confused():
    # `/rag-eval/api/...` 不能被页面路由吞掉
    resp = _client().get("/rag-eval/api/latest")
    assert resp.headers["content-type"].startswith("application/json")
