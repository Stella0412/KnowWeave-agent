"""意图成本账 CLI：按意图汇总 token 与成本，产物落 data/cost_report/latest.json。"""

import argparse
import json
import sys

from app.observability.cost_report import REPORT_PATH, CostReportError, refresh


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按意图汇总 Langfuse token 与成本")
    parser.add_argument("--days", type=int, default=7, help="统计窗口（天），默认 7")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    try:
        report = refresh(args.days)
    except CostReportError as exc:
        print(f"重跑失败：{exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"参数不对：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(
        {
            "product": str(REPORT_PATH),
            "window_days": report["window_days"],
            "totals": report["totals"],
            "top_intent": report["top_intent"],
            "warnings": report["warnings"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
