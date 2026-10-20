"""裁判回归：拿人工标过的固定判例，连跑多轮，检查裁判是否稳定且和人工一致。

判例在 `eval/judge_cases.jsonl`，每条都带人工期望（`expected_faithful`）和一句标注理由
（`note`）。这个脚本只调用裁判模型，不跑检索、不生成答案，所以几分钟就能跑完。

用法：
    .venv/Scripts/python.exe -m eval.judge_regression
    .venv/Scripts/python.exe -m eval.judge_regression --rounds 4
    .venv/Scripts/python.exe -m eval.judge_regression --only CASE_ --verbose

退出码：0 = 每轮全对且每条的预测跨轮稳定；1 = 有误判或不稳定；2 = 调用异常。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from app.config import settings
from app.core.llm import build_structured_model
from app.prompts.rag import FaithJudgeOutput, faith_judge_messages

CASES_PATH = Path(__file__).with_name("judge_cases.jsonl")


def load_cases(path: Path = CASES_PATH, only: str | None = None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if only:
        rows = [row for row in rows if only in row["id"]]
    return rows


def label(value: bool) -> str:
    return "忠实" if value else "编造"


def run_rounds(cases: list[dict], rounds: int) -> dict[str, list[bool]]:
    judge_model = build_structured_model(settings, FaithJudgeOutput, temperature=0.0)
    predictions: dict[str, list[bool]] = {case["id"]: [] for case in cases}
    for round_no in range(1, rounds + 1):
        print(f"第 {round_no}/{rounds} 轮…", file=sys.stderr, flush=True)
        for case in cases:
            verdict = judge_model.invoke(
                faith_judge_messages(case["query"], case["answer"], case["evidence"])
            )
            predictions[case["id"]].append(bool(verdict.faithful))
            if verdict.reason and verdict.reason.strip():
                case.setdefault("_reasons", []).append(verdict.reason.strip())
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="忠实度裁判回归")
    parser.add_argument("--rounds", type=int, default=4, help="连跑几轮（默认 4）")
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="判例 JSONL 路径")
    parser.add_argument("--only", default=None, help="只跑 id 里含这个子串的判例")
    parser.add_argument("--verbose", action="store_true", help="打印误判个案的裁判理由")
    args = parser.parse_args()

    cases = load_cases(args.cases, args.only)
    if not cases:
        print("没有可跑的判例。", file=sys.stderr)
        return 2

    try:
        predictions = run_rounds(cases, args.rounds)
    except Exception as exc:  # noqa: BLE001 - 回归脚本要把异常变成可读的退出码
        print(f"裁判调用失败：{exc}", file=sys.stderr)
        return 2

    expected = {case["id"]: bool(case["expected_faithful"]) for case in cases}
    mismatches = []
    print(f"\n裁判回归：{len(cases)} 条，跑 {args.rounds} 轮，裁判模型 {settings.llm_model}\n")
    print(f"{'个案':<9}{'人工期望':<8}{'裁判各轮':<28}{'稳定':<6}{'一致':<6}")
    for case in cases:
        preds = predictions[case["id"]]
        stable = len(set(preds)) == 1
        agree = all(pred == expected[case["id"]] for pred in preds)
        if not agree:
            mismatches.append(case)
        shown = "/".join(label(pred) for pred in preds)
        print(
            f"{case['id']:<9}{label(expected[case['id']]):<8}{shown:<28}"
            f"{'是' if stable else '否':<6}{'是' if agree else '否':<6}"
        )

    total_checks = len(cases) * args.rounds
    round_agreement = []
    for round_idx in range(args.rounds):
        hits = sum(
            1
            for case in cases
            if predictions[case["id"]][round_idx] == expected[case["id"]]
        )
        round_agreement.append(hits)
    agree_total = sum(round_agreement)
    stable_total = sum(1 for case in cases if len(set(predictions[case["id"]])) == 1)

    print(
        f"\n一致 {agree_total}/{total_checks} ({agree_total / total_checks:.1%})；"
        f"稳定 {stable_total}/{len(cases)}"
    )
    print("各轮一致：" + "、".join(f"{hits}/{len(cases)}" for hits in round_agreement))

    first_round = {
        case["id"]: predictions[case["id"]][0]
        for case in cases
        if predictions[case["id"]][0] != expected[case["id"]]
    }
    false_positives = [
        cid for cid, pred in first_round.items() if pred is False and expected[cid] is True
    ]
    false_negatives = [
        cid for cid, pred in first_round.items() if pred is True and expected[cid] is False
    ]
    print(
        f"首轮误报（人工忠实、裁判编造）：{len(false_positives)} 条"
        + (f"（{', '.join(false_positives)}）" if false_positives else "")
    )
    print(
        f"首轮漏报（人工编造、裁判忠实）：{len(false_negatives)} 条"
        + (f"（{', '.join(false_negatives)}）" if false_negatives else "")
    )

    if args.verbose and mismatches:
        print("\n误判个案：")
        for case in mismatches:
            reasons = case.get("_reasons") or []
            print(f"\n- {case['id']}（人工期望：{label(expected[case['id']])}）")
            print(f"  标注理由：{case['note']}")
            if reasons:
                print(f"  裁判理由：{reasons[0]}")

    ok = agree_total == total_checks and stable_total == len(cases)
    if not ok:
        print("\n回归未通过：有误判或不稳定。")
        return 1
    print("\n回归通过：每轮全对，且跨轮稳定。")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
