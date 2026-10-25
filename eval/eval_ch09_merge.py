"""ch09 语义阈值评估：只读固定 60 对，扫描 0.85–0.98。"""

import json
import math
from pathlib import Path


def load_pairs(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 50:
        raise ValueError("ch09_merge_pairs.jsonl 少于 50 对，拒绝评估")
    for row in rows:
        if set(row) != {"a", "b", "label"} or row["label"] not in {"dup", "distinct"}:
            raise ValueError(f"非法评估行: {row!r}")
    return rows


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def scan_thresholds(pairs, vectors, thresholds):
    out = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for pair, vector in zip(pairs, vectors):
            predicted = cosine(vector[0], vector[1]) >= threshold
            actual = pair["label"] == "dup"
            tp += predicted and actual
            fp += predicted and not actual
            fn += (not predicted) and actual
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append({"threshold": threshold, "precision": precision,
                    "recall": recall, "f1": f1})
    return out


def recommend_threshold(rows, min_precision=0.95):
    """spec §6.2：precision ≥ min_precision 下召回最高的阈值；无合格阈值返回 None。"""
    eligible = [row for row in rows if row["precision"] >= min_precision]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item["recall"], item["threshold"]))


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.core.embeddings import embed_texts

    pairs = load_pairs(Path(__file__).resolve().parent / "ch09_merge_pairs.jsonl")
    flat = embed_texts([text for pair in pairs for text in (pair["a"], pair["b"])])
    vectors = [flat[index:index + 2] for index in range(0, len(flat), 2)]
    rows = scan_thresholds(
        pairs, vectors,
        [round(0.85 + index * 0.01, 2) for index in range(14)],
    )
    for row in rows:
        print(
            f"threshold={row['threshold']:.2f} "
            f"precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1={row['f1']:.3f}"
        )
    best = recommend_threshold(rows)
    if best is not None:
        print(
            f"best={best['threshold']:.2f} "
            f"precision={best['precision']:.3f} recall={best['recall']:.3f} "
            f"f1={best['f1']:.3f}"
        )
    else:
        print("best=None（无 precision≥0.95 的阈值）")


if __name__ == "__main__":
    main()
