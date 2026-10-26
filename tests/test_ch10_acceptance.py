import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import acceptance as acceptance_router
from app.services.acceptance_service import AcceptanceService


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path):
    dataset = tmp_path / "dataset"
    _write(dataset / "manifest.json", {
        "splits": {"train": {"rows": 10}, "val": {"rows": 2}, "test": {"rows": 2}},
        "totals": {"labels": 17, "multi_label": 3},
    })
    model = tmp_path / "model"
    run = model / "run-1"
    _write(model / "latest.txt", "run-1")
    _write(run / "manifest.json", {"run": "run-1", "model": "hfl/chinese-roberta-wwm-ext"})
    _write(run / "thresholds.json", {"thresholds": {"退换货": 0.4}})
    _write(model / "current" / "export_manifest.json", {
        "opset": 17, "max_abs_diff": 0.0001,
    })
    names = [
        item["name"]
        for item in json.loads(
            Path("config/ch10_taxonomy.json").read_text(encoding="utf-8")
        )["labels"]
    ]
    per_class = {name: {"support": 0, "f1": 0.0} for name in names}
    per_class["退换货"] = {"support": 1, "f1": 0.8}
    evaldir = tmp_path / "eval"
    _write(evaldir / "metrics.json", {
        "metrics": {
            "total": 2, "macro_f1": 0.8, "micro_f1": 0.9, "exact_match": 0.5,
            "labels": per_class,
        },
        "real_slice": [], "golden": [{"text": "买大了想退"}],
        "meta": {"generated_at": "2026-10-27T10:00:00", "golden_pass": 1, "golden_total": 1},
    })
    (evaldir / "errors_to_audit.csv").write_text(
        "id,text,gold,pred,error_type,label,verdict,notes\n"
        "1,买大了,尺码,其他,FN,尺码,,\n"
        "2,发票,发票,其他,FN,发票,,\n",
        encoding="utf-8",
    )
    return AcceptanceService(
        eval_dir=evaldir, model_dir=model, dataset_dir=dataset,
    )


def test_overview_reports_metrics_and_checks(tmp_path):
    overview = _fixture(tmp_path).overview()
    assert overview["ready"] is True
    assert overview["test"]["macro_f1"] == 0.8
    assert overview["model"]["run"] == "run-1"
    checks = {item["name"]: item["ok"] for item in overview["checks"]}
    assert checks["测试集指标"] is True
    assert checks["ONNX 对齐"] is True


def test_eval_detail_includes_per_class(tmp_path):
    detail = _fixture(tmp_path).eval_detail()
    assert detail["metrics"]["labels"]["退换货"]["f1"] == 0.8
    assert detail["thresholds"]["退换货"] == 0.4


def test_errors_are_limited(tmp_path):
    service = _fixture(tmp_path)
    assert len(service.errors(limit=1)) == 1
    assert service.errors(limit=10)[0]["label"] == "尺码"


def test_acceptance_api_routes(tmp_path):
    mini = FastAPI()
    mini.state.acceptance = _fixture(tmp_path)
    mini.include_router(acceptance_router.router)
    client = TestClient(mini)
    assert client.get("/acceptance/api/overview").status_code == 200
    assert client.get("/acceptance/api/eval").json()["metrics"]["macro_f1"] == 0.8
    assert client.get("/acceptance/api/errors?limit=1").status_code == 200
