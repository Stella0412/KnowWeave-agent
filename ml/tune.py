"""在验证集上按类搜索阈值，写入训练 run 的 thresholds.json。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_model_dir(value: str) -> Path:
    path = Path(value)
    if path.suffix == ".txt" and path.exists():
        return path.parent / path.read_text(encoding="utf-8").strip() / "best"
    return path


def tune(model_dir, val_path, out_path, *, batch_size: int = 32) -> dict[str, float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ml.dataset_audit import read_jsonl
    from ml.decision import tune_thresholds
    from ml.taxonomy import load_taxonomy

    taxonomy = load_taxonomy()
    rows = read_jsonl(val_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch], max_length=128, truncation=True,
                padding=True, return_tensors="pt",
            )
            scores.extend(torch.sigmoid(model(**encoded).logits).cpu().numpy().tolist())
    gold = [taxonomy.multi_hot(row["labels"]) for row in rows]
    thresholds = tune_thresholds(scores, gold, taxonomy.names)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"thresholds": thresholds}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return thresholds


def main():
    parser = argparse.ArgumentParser(description="ch10 验证集阈值调优")
    parser.add_argument("--model-dir", default="data/ch10/model/latest.txt")
    parser.add_argument("--val", default="ch10-dataset/val.jsonl")
    parser.add_argument("--out", default="data/ch10/model/thresholds.json")
    args = parser.parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    out = Path(args.out)
    if args.out == "data/ch10/model/thresholds.json":
        out = model_dir.parent / "thresholds.json"
    print(json.dumps(tune(model_dir, args.val, out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
