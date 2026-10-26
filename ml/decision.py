from __future__ import annotations


def decide(scores, thresholds, names) -> list[str]:
    if len(scores) != len(thresholds) or len(scores) != len(names):
        raise ValueError("scores/thresholds/names 长度不一致")
    chosen = [name for name, score, threshold in zip(names, scores, thresholds) if score >= threshold]
    if not chosen:
        best = max(range(len(scores)), key=lambda i: (scores[i], -i))
        chosen = [names[best]]
    return [name for name in names if name in chosen]


def tune_thresholds(scores, gold, names, grid=None) -> dict[str, float]:
    grid = grid or [round(0.20 + 0.05 * i, 2) for i in range(11)]
    out = {}
    for col, name in enumerate(names):
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in grid:
            tp = fp = fn = 0
            for row_scores, row_gold in zip(scores, gold):
                pred = row_scores[col] >= threshold
                truth = bool(row_gold[col])
                tp += int(pred and truth)
                fp += int(pred and not truth)
                fn += int((not pred) and truth)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
                best_f1, best_threshold = f1, threshold
        out[name] = best_threshold
    return out
