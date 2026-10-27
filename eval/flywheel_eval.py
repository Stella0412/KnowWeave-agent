"""飞轮评估：定期在 ch04 评估集上重跑，并把每轮结果记账，供页面看趋势。

- 第一轮 `make eval-flywheel`（TRIGGER 默认「手动」）；
- 之后 `TRIGGER=定时 make eval-flywheel`，跑完自动和同一评估集的上轮对比；
- **只有同一评估集的两轮才比得动**：评估集文件内容一变（哈希不同），本轮重置基线；
- Faithfulness 掉出容差就写警告 —— 飞轮持续往库里灌知识，脏知识最先拉低的就是它。

每轮追加一行到 `data/eval_trends/rounds.jsonl`，评估结果本身照旧落
`data/rag_eval/latest.json`（页面「最近一轮」区块直接复用）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLES = Path("eval/samples_ch04.jsonl")
ROUNDS_PATH = Path("data/eval_trends/rounds.jsonl")
EVAL_SET = "ch04"
TRIGGERS = ("手动", "定时")
DEFAULT_TRIGGER = "手动"
MODES = ("full", "retrieval")

# 掉多少算「劣化」：Faithfulness 最敏感，门槛设得最低。
WATCH = {
    "faithfulness": 0.03,
    "coverage": 0.03,
    "refusal_accuracy": 0.02,
    "recall_at_5": 0.02,
}
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "coverage": "答案覆盖",
    "refusal_accuracy": "拒答正确率",
    "misblocked_rate": "误拦率",
    "recall_at_5": "Recall@5",
    "mrr_at_10": "MRR@10",
}


def eval_set_identity(path: Path = SAMPLES) -> tuple[str, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return EVAL_SET, digest


def _ratio(numerator, denominator):
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def metrics_from_result(result: dict) -> dict:
    answers = result.get("answers") or {}
    chosen = (result.get("strategies") or {}).get("hybrid_rerank") or {}
    return {
        "faithfulness": _ratio(answers.get("faithful"), answers.get("faithful_total")),
        "coverage": round(answers["coverage"], 4) if answers.get("answered") else None,
        "refusal_accuracy": _ratio(answers.get("refusal_correct"), answers.get("refusal_total")),
        "misblocked_rate": _ratio(
            answers.get("refused_answerable"), answers.get("answerable_total")
        ),
        "recall_at_5": chosen.get("recall@5"),
        "mrr_at_10": chosen.get("mrr@10"),
    }


def compare(previous: dict | None, current: dict) -> dict:
    """只能和同一评估集（同一内容哈希）的上一轮比。"""
    if previous is None:
        return {"status": "baseline", "previous_round_id": None, "deltas": {}, "warnings": []}
    if previous.get("eval_set_hash") != current["eval_set_hash"]:
        return {
            "status": "incomparable",
            "previous_round_id": previous.get("round_id"),
            "reason": (
                f"评估集内容已变（{previous.get('eval_set_hash')} → {current['eval_set_hash']}），"
                "本轮重置基线"
            ),
            "deltas": {},
            "warnings": [],
        }
    deltas = {}
    warnings = []
    for metric in METRIC_LABELS:
        now_value = current["metrics"].get(metric)
        before = (previous.get("metrics") or {}).get(metric)
        if now_value is None or before is None:
            continue
        delta = round(now_value - before, 4)
        deltas[metric] = delta
        limit = WATCH.get(metric)
        if limit is not None and delta <= -limit:
            warnings.append(
                f"{METRIC_LABELS[metric]} 下降 {abs(delta):.3f}"
                f"（{before:.3f} → {now_value:.3f}），脏知识可能已进库"
            )
    return {
        "status": "comparable",
        "previous_round_id": previous.get("round_id"),
        "deltas": deltas,
        "warnings": warnings,
    }


def last_round_of_set(rounds: list[dict], eval_set: str) -> dict | None:
    for round in reversed(rounds):
        if round.get("eval_set") == eval_set:
            return round
    return None


def build_round(
    result: dict,
    *,
    previous: dict | None,
    trigger: str,
    eval_set: str,
    eval_set_hash: str,
    round_id: str,
    finished_at: str,
    mode: str = "full",
) -> dict:
    current = {
        "round_id": round_id,
        "trigger": trigger,
        "mode": mode,
        "eval_set": eval_set,
        "eval_set_hash": eval_set_hash,
        "sample_count": result.get("samples"),
        "collection_count": result.get("collection_count"),
        "degraded": result.get("degraded"),
        "started_at": result.get("started_at"),
        "finished_at": finished_at,
        "metrics": metrics_from_result(result),
    }
    current["comparison"] = compare(previous, current)
    return current


def default_runner(run_id: str, mode: str):
    from app.services import rag_eval_service

    result = rag_eval_service.run_evaluation(mode=mode, run_id=run_id)
    rag_eval_service.persist_result(result)
    return result


def append_round(round: dict, path: Path = ROUNDS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(round, ensure_ascii=False) + "\n")


def read_rounds(path: Path = ROUNDS_PATH) -> list[dict]:
    if not path.exists():
        return []
    rounds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rounds.append(json.loads(line))
        except ValueError:
            continue
    return rounds


def run(
    *,
    trigger: str = DEFAULT_TRIGGER,
    mode: str = "full",
    runner=default_runner,
    samples_path: Path = SAMPLES,
    rounds_path: Path = ROUNDS_PATH,
    now: datetime | None = None,
) -> dict:
    if trigger not in TRIGGERS:
        raise ValueError(f"未知触发方式：{trigger}（可选：{'/'.join(TRIGGERS)}）")
    if mode not in MODES:
        raise ValueError(f"未知评估模式：{mode}（可选：{'/'.join(MODES)}）")

    eval_set, eval_set_hash = eval_set_identity(samples_path)
    # round_id 用开跑时刻；finished_at 必须等跑完再取，否则趋势表按完成时间排序会错位。
    stamp = now or datetime.now()
    round_id = stamp.strftime("%Y%m%d-%H%M%S") + "-flywheel"
    result = runner(round_id, mode)
    previous = last_round_of_set(read_rounds(rounds_path), eval_set)
    round = build_round(
        result,
        previous=previous,
        trigger=trigger,
        eval_set=eval_set,
        eval_set_hash=eval_set_hash,
        round_id=round_id,
        finished_at=(now or datetime.now()).isoformat(timespec="seconds"),
        mode=mode,
    )
    append_round(round, rounds_path)
    return round


def main(argv: list[str] | None = None) -> int:
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="飞轮评估：重跑 ch04 并记入趋势")
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument(
        "--trigger", default=os.environ.get("TRIGGER", DEFAULT_TRIGGER),
        help="手动 / 定时（也可用环境变量 TRIGGER）",
    )
    args = parser.parse_args(argv)
    try:
        round = run(trigger=args.trigger, mode=args.mode)
    except ValueError as e:
        print(f"参数错误：{e}", file=sys.stderr)
        return 2
    print(json.dumps(round, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
