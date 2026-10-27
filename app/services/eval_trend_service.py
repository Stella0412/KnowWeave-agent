"""飞轮评估趋势：只读 `data/eval_trends/rounds.jsonl`，按评估集分行。

同一评估集（同一内容哈希）的多轮才放在同一行、才能比；评估集改版就另起一行，
免得把两套题的数字放在一起看。
"""

from __future__ import annotations

from pathlib import Path

from eval.flywheel_eval import read_rounds

MAX_ROUNDS_PER_SET = 6
TREND_PATH = Path("data/eval_trends/rounds.jsonl")


def grouped(rounds: list[dict], max_rounds: int = MAX_ROUNDS_PER_SET) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for round in rounds:
        key = (round.get("eval_set") or "unknown", round.get("eval_set_hash") or "")
        groups.setdefault(key, []).append(round)
    sets = [
        {"eval_set": key[0], "eval_set_hash": key[1], "rounds": rows[-max_rounds:]}
        for key, rows in groups.items()
    ]
    sets.sort(key=lambda item: item["rounds"][-1].get("finished_at") or "", reverse=True)
    return sets


def latest_warnings(rounds: list[dict]) -> list[str]:
    if not rounds:
        return []
    comparison = rounds[-1].get("comparison") or {}
    return list(comparison.get("warnings") or [])


def overview(path: Path = TREND_PATH) -> dict:
    rounds = read_rounds(path)
    latest = rounds[-1] if rounds else None
    return {
        "rounds": rounds,
        "sets": grouped(rounds),
        "latest_round_id": latest.get("round_id") if latest else None,
        "latest_trigger": latest.get("trigger") if latest else None,
        "warnings": latest_warnings(rounds),
    }
