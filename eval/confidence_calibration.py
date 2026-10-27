"""置信度阈值校准：在 ch04 评估集（240 可答 + 60 应拒）上实跑，扫一遍阈值。

权重在 `app/rag/confidence.py` 里是常量，**校准只决定线划在哪里**：
把可答题的置信度分布和应拒题的分布各算一遍，再扫 0.05→0.95，看哪条线把两拨分得最开
（Youden J = 可答通过率 − 应拒放行率）。推荐值是数学最优点，在用值是可接受的误放率下的人工选择。

产物 `data/calibration/latest.json`：页面只读产物；重跑走评估页按钮或
`make calibrate-confidence`。
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.confidence import ConfidenceSignals, signals

SAMPLES = Path("eval/samples_ch04.jsonl")
REPORT_PATH = Path("data/calibration/latest.json")
SCAN_START, SCAN_STOP, SCAN_STEP = 0.05, 0.95, 0.01

ProgressFn = Callable[[int, int], None]


def load_samples(path: Path = SAMPLES) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], p: float) -> float | None:
    """最近秩法，与观测面板 P50/P95 同一口径。"""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": round(min(values), 4) if values else None,
        "p25": round(percentile(values, 0.25), 4) if values else None,
        "median": round(percentile(values, 0.5), 4) if values else None,
        "p75": round(percentile(values, 0.75), 4) if values else None,
        "max": round(max(values), 4) if values else None,
    }


def thresholds() -> list[float]:
    steps = int(round((SCAN_STOP - SCAN_START) / SCAN_STEP)) + 1
    return [round(SCAN_START + index * SCAN_STEP, 2) for index in range(steps)]


def _rate(values: list[float], passed: Callable[[float], bool]) -> float:
    return sum(1 for value in values if passed(value)) / len(values) if values else 0.0


def scan(answerable: list[float], refuse: list[float]) -> dict:
    points = thresholds()
    pass_rates = [round(_rate(answerable, lambda score, t=t: score >= t), 4) for t in points]
    false_pass_rates = [
        round(_rate(refuse, lambda score, t=t: score >= t), 4) for t in points
    ]
    return {
        "start": SCAN_START, "stop": SCAN_STOP, "step": SCAN_STEP,
        "thresholds": points,
        "pass_rates": pass_rates,
        "false_pass_rates": false_pass_rates,
    }


def recommend(scan_result: dict) -> tuple[float, float]:
    """Youden J 最大的那条线；同 J 取更高阈值（同样分得开时选更保守的）。"""
    best_t = SCAN_START
    best_j = -1.0
    for t, passed, false_passed in zip(
        scan_result["thresholds"], scan_result["pass_rates"], scan_result["false_pass_rates"]
    ):
        youden = round(passed - false_passed, 4)
        if youden >= best_j:
            best_j, best_t = youden, t
    return best_t, best_j


def at_threshold(records: list[dict], t: float) -> dict:
    answerable = [row for row in records if not row["should_refuse"]]
    refuse = [row for row in records if row["should_refuse"]]
    misblocked = [row for row in answerable if row["score"] < t]
    passed = [row for row in refuse if row["score"] >= t]
    return {
        "threshold": t,
        "pass_rate": round((len(answerable) - len(misblocked)) / len(answerable), 4)
        if answerable else 0.0,
        "false_pass_rate": round(len(passed) / len(refuse), 4) if refuse else 0.0,
        "misblocked_count": len(misblocked),
        "misblocked_ids": [row["id"] for row in misblocked],
        "false_pass_count": len(passed),
        "false_pass_ids": [row["id"] for row in passed],
    }


def _examples(rows: list[dict], limit: int = 5, *, reverse: bool = False) -> list[dict]:
    # 误拦看「分数最低」的（最该被捞回来）；误放看「分数最高」的（最自信地答错）。
    picked = sorted(rows, key=lambda row: row["score"], reverse=reverse)
    return [
        {"id": row["id"], "query": row["query"], "score": row["score"],
         "top1": row.get("top1"), "bucket": row.get("bucket")}
        for row in picked[:limit]
    ]


def summarize(
    records: list[dict],
    *,
    in_use: float,
    generated_at: str,
    scan_result: dict | None = None,
) -> dict:
    # 精排降级时没有精排分，置信度不可比（线上这时也不套闸门），不参与分布与扫描。
    usable = [row for row in records if not row.get("degraded")]
    answerable = [row["score"] for row in usable if not row["should_refuse"]]
    refuse = [row["score"] for row in usable if row["should_refuse"]]
    scan_result = scan_result or scan(answerable, refuse)
    recommended, youden = recommend(scan_result)
    at_use = at_threshold(usable, in_use)
    at_best = at_threshold(usable, recommended)
    return {
        "generated_at": generated_at,
        "in_use": in_use,
        "recommended": recommended,
        "youden": youden,
        "sample_count": len(usable),
        "degraded_count": len(records) - len(usable),
        "answerable_count": len(answerable),
        "refuse_count": len(refuse),
        "distribution": {
            "answerable": distribution(answerable),
            "refuse": distribution(refuse),
        },
        "scan": scan_result,
        "at_in_use": at_use,
        "at_recommended": at_best,
        "misblocked_examples": _examples(
            [row for row in usable if not row["should_refuse"] and row["score"] < in_use]
        ),
        "false_pass_examples": _examples(
            [row for row in usable if row["should_refuse"] and row["score"] >= in_use],
            reverse=True,
        ),
        "degraded": [row["id"] for row in records if row.get("degraded")],
    }


def diagnostic_retrieve(sample: dict, source_files: dict | None = None):
    """按线上策略跑一遍检索，但把闸门放到 0 并跳过自评：拿到每题的证据与置信度。"""
    from app.rag.pipeline import default_source_files, run_pipeline

    result = run_pipeline(
        sample["query"],
        source_files=source_files if source_files is not None else default_source_files(),
        confidence_min_score=0.0,
        enable_self_check=False,
    )
    return result, signals(result.items, sample["query"])


def collect(
    samples: list[dict],
    *,
    limit: int | None = None,
    progress: ProgressFn | None = None,
    retrieve: Callable[[dict], tuple[Any, ConfidenceSignals]] | None = None,
    source_files: dict | None = None,
) -> list[dict]:
    """逐题跑线上检索链路，记录四信号。`retrieve` 可注入（测试不碰真模型）。"""
    chosen = samples[:limit] if limit is not None else samples
    if retrieve is None:
        files = source_files if source_files is not None else _default_source_files()
        retrieve = lambda sample: diagnostic_retrieve(sample, files)  # noqa: E731
    records = []
    total = len(chosen)
    for index, sample in enumerate(chosen, start=1):
        result, confidence = retrieve(sample)
        records.append({
            "id": sample["id"],
            "bucket": sample["bucket"],
            "query": sample["query"],
            "should_refuse": bool(sample.get("should_refuse")),
            "score": round(confidence.score, 4),
            "top1": round(confidence.top1, 4),
            "effective": round(confidence.effective, 4),
            "margin": round(confidence.margin, 4),
            "clause": round(confidence.clause, 4),
            "degraded": bool(getattr(result, "rerank_degraded", False)),
        })
        if progress is not None:
            progress(index, total)
    return records


def _default_source_files() -> dict:
    from app.rag.pipeline import default_source_files

    return default_source_files()


def write_report(report: dict, path: Path | None = None) -> Path:
    path = path or REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_report(path: Path | None = None) -> dict | None:
    path = path or REPORT_PATH
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None
    if isinstance(report, dict):
        # 旧产物没有 degraded_count 字段：可从明细补出来，页面不必兼容两套。
        report.setdefault("degraded_count", len(report.get("degraded") or []))
    return report


def run(
    *,
    limit: int | None = None,
    progress: ProgressFn | None = None,
    retrieve: Callable[[dict], tuple[Any, ConfidenceSignals]] | None = None,
    in_use: float | None = None,
    path: Path | None = None,
    samples: list[dict] | None = None,
) -> dict:
    from app.config import settings

    chosen_samples = samples if samples is not None else load_samples()
    records = collect(
        chosen_samples, limit=limit, progress=progress, retrieve=retrieve,
    )
    report = summarize(
        records,
        in_use=settings.confidence_min_score if in_use is None else in_use,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    write_report(report, path)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="ch04 评估集上的置信度阈值校准")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟用）")
    args = parser.parse_args(argv)

    def show(done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            print(f"已跑 {done}/{total} 题", flush=True)

    report = run(limit=args.limit, progress=show)
    print(json.dumps({
        "product": str(REPORT_PATH),
        "sample_count": report["sample_count"],
        "in_use": report["in_use"],
        "recommended": report["recommended"],
        "youden": report["youden"],
        "at_in_use": {
            key: report["at_in_use"][key]
            for key in ("pass_rate", "false_pass_rate", "misblocked_count", "false_pass_count")
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
