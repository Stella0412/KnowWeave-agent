from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from ml.dataset_audit import read_jsonl
from ml.decision import decide
from ml.taxonomy import load_taxonomy


def confusion_2x2(gold_bin, pred_bin) -> dict[str, int]:
    tp = sum(1 for g, p in zip(gold_bin, pred_bin) if g and p)
    fp = sum(1 for g, p in zip(gold_bin, pred_bin) if (not g) and p)
    fn = sum(1 for g, p in zip(gold_bin, pred_bin) if g and (not p))
    tn = sum(1 for g, p in zip(gold_bin, pred_bin) if (not g) and (not p))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics(gold, pred, names) -> dict:
    labels = {}
    for name in names:
        g = [name in row for row in gold]
        p = [name in row for row in pred]
        cell = confusion_2x2(g, p)
        precision = cell["tp"] / (cell["tp"] + cell["fp"]) if cell["tp"] + cell["fp"] else 0.0
        recall = cell["tp"] / (cell["tp"] + cell["fn"]) if cell["tp"] + cell["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        labels[name] = {
            **cell,
            "support": sum(g),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    macro_f1 = sum(item["f1"] for item in labels.values()) / len(labels)
    total_tp = sum(item["tp"] for item in labels.values())
    total_fp = sum(item["fp"] for item in labels.values())
    total_fn = sum(item["fn"] for item in labels.values())
    micro_f1 = 2 * total_tp / max(1, 2 * total_tp + total_fp + total_fn)
    exact = sum(1 for g, p in zip(gold, pred) if set(g) == set(p)) / max(1, len(gold))
    return {
        "labels": labels,
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "exact_match": round(exact, 4),
        "avg_labels": round(sum(len(p) for p in pred) / max(1, len(pred)), 2),
        "total": len(gold),
    }


def export_error_cases(rows, gold, pred, out_path, names, per_class=2) -> int:
    quota = Counter()
    picked = []
    for index, (row, g, p) in enumerate(zip(rows, gold, pred), 1):
        fp_labels = [name for name in p if name not in g]
        fn_labels = [name for name in g if name not in p]
        for kind, labels in (("FP", fp_labels), ("FN", fn_labels)):
            for label in labels:
                if quota[(label, kind)] >= per_class:
                    continue
                quota[(label, kind)] += 1
                picked.append({
                    "id": row.get("id") or f"row-{index}",
                    "text": row["text"],
                    "gold": "|".join(g),
                    "pred": "|".join(p),
                    "error_type": kind,
                    "label": label,
                })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["id", "text", "gold", "pred", "error_type", "label", "verdict", "notes"],
        )
        writer.writeheader()
        for item in picked:
            writer.writerow({**item, "verdict": "", "notes": ""})
    return len(picked)


def _load_thresholds(model_dir, names) -> list[float]:
    for candidate in (Path(model_dir).parent / "thresholds.json",
                      Path(model_dir) / "thresholds.json"):
        if candidate.exists():
            raw = json.loads(candidate.read_text(encoding="utf-8")).get("thresholds") or {}
            return [float(raw.get(name, 0.5)) for name in names]
    return [0.5] * len(names)


def _predict(rows, model_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    taxonomy = load_taxonomy()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    names = list(taxonomy.names)
    thresholds = _load_thresholds(model_dir, names)
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            encoded = tokenizer([row["text"] for row in batch], max_length=128,
                                truncation=True, padding=True, return_tensors="pt")
            logits = model(**encoded).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            for row_scores in probs:
                out.append(decide(list(map(float, row_scores)), thresholds, names))
    return out


def render_report(metrics, real_rows, golden_rows, meta) -> str:
    lines = [
        f"# ch10 主题分类器评测报告（{meta.get('generated_at')}）", "",
        f"- 模型：`{meta.get('model_dir')}`",
        f"- 数据：`{meta.get('test_path')}`",
        f"- 测试集：{metrics['total']} 条（每类 support 9–14，单类指标误差较大）",
        f"- macro-F1：{metrics['macro_f1']}；micro-F1：{metrics['micro_f1']}；完全匹配率：{metrics['exact_match']}",
        "", "## 每类指标与 2×2 混淆矩阵", "",
        "| 类目 | support | TP | FP | FN | TN | P | R | F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, item in metrics["labels"].items():
        lines.append(
            f"| {name} | {item['support']} | {item['tp']} | {item['fp']} | {item['fn']} | "
            f"{item['tn']} | {item['precision']} | {item['recall']} | {item['f1']} |"
        )
    lines += ["", "## 真实域切片（单列，不参与核心 F1）", ""]
    if real_rows:
        lines += ["| question_id | 原文 | 真实标签 | 预测标签 | 对错 |", "|---|---|---|---|---|"]
        for item in real_rows:
            lines.append(
                f"| {item['question_id']} | {item['text']} | {'/'.join(item['gold'])} | "
                f"{'/'.join(item['pred'])} | {'✅' if set(item['gold']) == set(item['pred']) else '❌'} |"
            )
    else:
        lines.append("本次未提供带标签的真实切片；真实池问题由旁路批量任务单独归类。")
    lines += ["", "## 黄金用例", "", "| 用例 | 期望 | 预测 | 结果 |", "|---|---|---|---|"]
    for item in golden_rows:
        lines.append(
            f"| {item['text']} | {'/'.join(item['expected'])} | {'/'.join(item['pred'])} | "
            f"{'✅' if set(item['expected']) == set(item['pred']) else '❌'} |"
        )
    lines += ["", "> 核心指标是提供的测试集估计；每类 support 较小，不能外推到全量线上分布。", ""]
    return "\n".join(lines)


def evaluate_model(model_dir, test_path, real_path, golden_path, out_dir, report_path) -> dict:
    taxonomy = load_taxonomy()
    test_rows = read_jsonl(test_path)
    for index, row in enumerate(test_rows, 1):
        row.setdefault("id", f"test-{index}")
    gold = [row["labels"] for row in test_rows]
    pred = _predict(test_rows, model_dir)
    metrics = compute_metrics(gold, pred, taxonomy.names)

    real_out = []
    if real_path and Path(real_path).exists():
        real_rows = read_jsonl(real_path)
        if real_rows:
            real_pred = _predict(real_rows, model_dir)
            for row, p in zip(real_rows, real_pred):
                real_out.append({
                    "question_id": row.get("question_id"),
                    "text": row["text"],
                    "gold": row.get("labels") or [],
                    "pred": p,
                })

    golden_out = []
    if golden_path and Path(golden_path).exists():
        golden = read_jsonl(golden_path)
        if golden:
            golden_pred = _predict(golden, model_dir)
            for row, p in zip(golden, golden_pred):
                golden_out.append({
                    "text": row["text"],
                    "expected": row["expected_labels"],
                    "pred": p,
                })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    error_count = export_error_cases(
        test_rows, gold, pred, out_dir / "errors_to_audit.csv", taxonomy.names,
    )
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "test_path": str(test_path),
        "golden_pass": sum(1 for item in golden_out if set(item["expected"]) == set(item["pred"])),
        "golden_total": len(golden_out),
        "error_samples": error_count,
    }
    payload = {"metrics": metrics, "real_slice": real_out, "golden": golden_out, "meta": meta}
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(metrics, real_out, golden_out, meta), encoding="utf-8")
    return payload
