from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


class AcceptanceService:
    """读取 ch10 训练/评测/导出产物，供验收页面展示；不触发训练或推理。"""

    def __init__(
        self,
        eval_dir="data/ch10/eval",
        model_dir="data/ch10/model",
        dataset_dir="ch10-dataset",
        report_path="eval/reports/ch10-topic-2026-10-26.md",
    ):
        self._eval_dir = Path(eval_dir)
        self._model_dir = Path(model_dir)
        self._dataset_dir = Path(dataset_dir)
        self._report_path = Path(report_path)

    @staticmethod
    def _read_json(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _run_dir(self) -> Path | None:
        latest = self._model_dir / "latest.txt"
        try:
            name = latest.read_text(encoding="utf-8").strip().strip('"')
        except OSError:
            return None
        return self._model_dir / name if name else None

    def _eval_payload(self) -> dict | None:
        return self._read_json(self._eval_dir / "metrics.json")

    def _thresholds(self) -> dict:
        run = self._run_dir()
        for candidate in (
            run / "thresholds.json" if run else None,
            self._model_dir / "current" / "thresholds.json",
        ):
            if candidate is not None and candidate.exists():
                raw = self._read_json(candidate) or {}
                return raw.get("thresholds") or {}
        return {}

    def _export_manifest(self) -> dict | None:
        run = self._run_dir()
        for candidate in (
            self._model_dir / "current" / "export_manifest.json",
            run / "onnx" / "export_manifest.json" if run else None,
        ):
            if candidate is not None and candidate.exists():
                return self._read_json(candidate)
        return None

    def overview(self) -> dict:
        payload = self._eval_payload()
        dataset = self._read_json(self._dataset_dir / "manifest.json") or {}
        run = self._run_dir()
        run_manifest = self._read_json(run / "manifest.json") if run else None
        export = self._export_manifest()
        metrics = (payload or {}).get("metrics") or {}
        meta = (payload or {}).get("meta") or {}
        golden = (payload or {}).get("golden") or []
        checks = [
            {
                "name": "测试集指标",
                "ok": bool(metrics) and len(metrics.get("labels") or {}) == 17,
                "detail": f"{metrics.get('total', 0)} 条 / 17 类" if metrics else "尚未评测",
            },
            {
                "name": "ONNX 对齐",
                "ok": bool(export) and float(export.get("max_abs_diff", 1)) < 1e-3,
                "detail": (
                    f"max|Δ|={export.get('max_abs_diff')} (opset {export.get('opset')})"
                    if export else "尚未导出"
                ),
            },
            {
                "name": "黄金用例",
                "ok": bool(golden) and meta.get("golden_total", 0) > 0,
                "detail": f"{meta.get('golden_pass', 0)}/{meta.get('golden_total', 0)}",
            },
            {
                "name": "数据集产物",
                "ok": bool(dataset.get("splits")),
                "detail": (
                    " / ".join(
                        f"{split} {dataset['splits'].get(split, {}).get('rows', 0)}"
                        for split in ("train", "val", "test")
                    ) if dataset.get("splits") else "缺少 manifest"
                ),
            },
        ]
        splits = dataset.get("splits") or {}
        raw_labels = (dataset.get("totals") or {}).get("labels", 0)
        label_count = len(raw_labels) if isinstance(raw_labels, dict) else int(raw_labels or 0)
        return {
            "ready": bool(payload),
            "generated_at": meta.get("generated_at"),
            "model": {
                "run": (run_manifest or {}).get("run") or (run.name if run else None),
                "base_model": (run_manifest or {}).get("model"),
                "trained_at": (run_manifest or {}).get("trained_at"),
                "exported": bool(export),
                "max_abs_diff": (export or {}).get("max_abs_diff"),
                "opset": (export or {}).get("opset"),
            },
            "dataset": {
                "train": splits.get("train", {}).get("rows", 0),
                "val": splits.get("val", {}).get("rows", 0),
                "test": splits.get("test", {}).get("rows", 0),
                "multi_label": (dataset.get("totals") or {}).get("multi_label", 0),
                "labels": label_count,
            },
            "test": {
                "total": metrics.get("total", 0),
                "macro_f1": metrics.get("macro_f1"),
                "micro_f1": metrics.get("micro_f1"),
                "exact_match": metrics.get("exact_match"),
            },
            "golden": {
                "passed": meta.get("golden_pass", 0),
                "total": meta.get("golden_total", 0),
            },
            "real_slice": len((payload or {}).get("real_slice") or []),
            "checks": checks,
        }

    def eval_detail(self) -> dict:
        payload = self._eval_payload() or {}
        return {
            "metrics": payload.get("metrics") or {},
            "meta": payload.get("meta") or {},
            "golden": payload.get("golden") or [],
            "real_slice": payload.get("real_slice") or [],
            "thresholds": self._thresholds(),
            "report_exists": self._report_path.exists(),
            "report_path": str(self._report_path),
        }

    def artifacts(self) -> list[dict]:
        run = self._run_dir()
        candidates = [
            ("训练集", self._dataset_dir / "train.jsonl"),
            ("验证集", self._dataset_dir / "val.jsonl"),
            ("测试集", self._dataset_dir / "test.jsonl"),
            ("数据 manifest", self._dataset_dir / "manifest.json"),
            ("评测结果", self._eval_dir / "metrics.json"),
            ("错例 CSV", self._eval_dir / "errors_to_audit.csv"),
            ("评测报告", self._report_path),
            ("训练 manifest", run / "manifest.json" if run else None),
            ("验证指标", run / "metrics.json" if run else None),
            ("阈值", run / "thresholds.json" if run else None),
            ("ONNX 模型", self._model_dir / "current" / "model.onnx"),
            ("导出 manifest", self._model_dir / "current" / "export_manifest.json"),
        ]
        out = []
        for name, path in candidates:
            if path is None:
                continue
            exists = path.exists()
            stat = path.stat() if exists else None
            out.append({
                "name": name,
                "path": str(path),
                "exists": exists,
                "bytes": stat.st_size if stat else 0,
                "modified_at": (
                    datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
                    if stat else None
                ),
            })
        return out

    def errors(self, limit: int = 50) -> list[dict]:
        path = self._eval_dir / "errors_to_audit.csv"
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return rows[:max(0, limit)]
