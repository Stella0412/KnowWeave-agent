from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback,
    Trainer, TrainingArguments, set_seed,
)

from ml.dataset import MultiLabelDataset
from ml.dataset_audit import load_split
from ml.taxonomy import load_taxonomy

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"


def compute_pos_weight(rows, taxonomy, cap: float = 10.0) -> torch.Tensor:
    positives = np.zeros(len(taxonomy.categories))
    for row in rows:
        for label in row["labels"]:
            positives[taxonomy.index(label)] += 1
    total = max(1, len(rows))
    negatives = total - positives
    weight = np.divide(negatives, np.maximum(positives, 1))
    weight = np.clip(weight, 1.0, cap)
    return torch.tensor(weight, dtype=torch.float)


def load_training_rows(train_path, val_path):
    return load_split(train_path), load_split(val_path)


class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs.logits, labels, pos_weight=self.pos_weight.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def _metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    gold = labels.astype(int)
    tp = (preds * gold).sum(axis=0)
    fp = (preds * (1 - gold)).sum(axis=0)
    fn = ((1 - preds) * gold).sum(axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp, dtype=float), where=(precision + recall) > 0)
    return {"macro_f1": float(f1.mean()), "micro_f1": float(
        (2 * tp.sum()) / max(1, 2 * tp.sum() + fp.sum() + fn.sum())
    )}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def train(train_path, val_path, output_dir, *, device="auto", epochs=4, limit=None,
          seed=42):
    taxonomy = load_taxonomy()
    train_rows, val_rows = load_training_rows(train_path, val_path)
    if limit:
        train_rows, val_rows = train_rows[:limit], val_rows[:max(1, limit // 5)]
    set_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(taxonomy.categories),
        problem_type="multi_label_classification",
        id2label={i: name for i, name in enumerate(taxonomy.names)},
        label2id={name: i for i, name in enumerate(taxonomy.names)},
    )
    run_dir = Path(output_dir) / f"run-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = math.ceil(len(train_rows) / (16 * 2))
    warmup_steps = max(1, int(steps_per_epoch * epochs * 0.1))
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        fp16=(device == "cuda"),
        use_cpu=(device == "cpu"),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to=[],
        seed=seed,
        dataloader_num_workers=0,
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=MultiLabelDataset(train_rows, tokenizer, taxonomy),
        eval_dataset=MultiLabelDataset(val_rows, tokenizer, taxonomy),
        compute_metrics=_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        pos_weight=compute_pos_weight(train_rows, taxonomy),
    )
    started = time.time()
    trainer.train()
    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    metrics = trainer.evaluate()
    (run_dir / "metrics.json").write_text(json.dumps(_jsonable({
        "eval": metrics,
        "best_epoch": getattr(trainer.state, "best_epoch", None),
        "elapsed_seconds": round(time.time() - started, 1),
        "device": device,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "label_map.json").write_text(json.dumps(
        {"labels": list(taxonomy.names)}, ensure_ascii=False, indent=2
    ), encoding="utf-8")
    (run_dir / "train_config.json").write_text(json.dumps({
        "model": MODEL_NAME, "epochs": epochs, "seed": seed, "max_length": 128,
        "batch_size": 16, "grad_accum": 2, "learning_rate": 2e-5,
        "train_path": str(train_path), "val_path": str(val_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "run": run_dir.name, "model": MODEL_NAME, "labels": list(taxonomy.names),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(output_dir) / "latest.txt").write_text(run_dir.name, encoding="utf-8")
    return {"run_dir": str(run_dir), "metrics": _jsonable(metrics)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="ch10-dataset/train.jsonl")
    parser.add_argument("--val", default="ch10-dataset/val.jsonl")
    parser.add_argument("--output-dir", default="data/ch10/model")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train(args.train, args.val, args.output_dir, device=args.device,
                   epochs=args.epochs, limit=args.limit, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
