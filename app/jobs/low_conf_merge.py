"""数据飞轮离线归并：指纹回填 → 精确归并 → 语义归并。

用法：
    uv run python -m app.jobs.low_conf_merge --dry-run
    uv run python -m app.jobs.low_conf_merge --threshold 0.93 --json
    uv run python -m app.jobs.low_conf_merge --skip-semantic
    uv run python -m app.jobs.low_conf_merge --require-semantic

阈值/条数/批大小默认从 Settings 解析（low_conf_merge_threshold / low_conf_merge_limit /
low_conf_merge_batch_size），命令行显式传值则覆盖。
"""

import argparse
import asyncio
import json
import sys

from app.config import Settings
from app.db.engine import SessionFactory
from app.services.flywheel_service import FlywheelService


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="低置信度池离线归并")
    parser.add_argument("--dry-run", action="store_true", help="只出报告，不写库")
    parser.add_argument("--threshold", type=float, default=None,
                        help="语义归并余弦阈值（默认取 Settings）")
    parser.add_argument("--limit", type=int, default=None,
                        help="语义归并候选条数上限（默认取 Settings）")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="embedding 批大小（默认取 Settings）")
    semantic = parser.add_mutually_exclusive_group()
    semantic.add_argument("--skip-semantic", action="store_true")
    semantic.add_argument("--require-semantic", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _resolve(args: argparse.Namespace, s: Settings) -> tuple[float, int, int]:
    return (
        args.threshold if args.threshold is not None else s.low_conf_merge_threshold,
        args.limit if args.limit is not None else s.low_conf_merge_limit,
        args.batch_size if args.batch_size is not None else s.low_conf_merge_batch_size,
    )


def _public(report: dict) -> dict:
    return {k: v for k, v in report.items() if not k.startswith("_")}


async def run(args: argparse.Namespace) -> int:
    service = FlywheelService(SessionFactory)
    threshold, limit, batch_size = _resolve(args, service._settings)
    backfilled = 0
    if not args.dry_run:
        backfilled = await service.backfill_fingerprints()
    exact = await service.merge_exact(dry_run=args.dry_run)
    semantic = None
    if not args.skip_semantic:
        semantic = await service.merge_semantic(
            threshold=threshold, limit=limit,
            batch_size=batch_size, dry_run=args.dry_run,
        )
    cluster_ids = set(exact.get("_cluster_ids") or [])
    if semantic:
        cluster_ids |= set(semantic.get("_cluster_ids") or [])
    report = {
        "threshold": threshold,
        "limit": limit,
        "batch_size": batch_size,
        "backfilled": backfilled,
        "exact_merged": exact["exact_merged"],
        "semantic_merged": semantic["semantic_merged"] if semantic else 0,
        "merged": exact["exact_merged"] + (semantic["semantic_merged"] if semantic else 0),
        "clusters": len(cluster_ids),
        "skipped": exact["skipped"] + (semantic["skipped"] if semantic else 0),
        "skipped_cross_status": exact["skipped_cross_status"] + (
            semantic["skipped_cross_status"] if semantic else 0
        ),
        "skipped_locked": exact["skipped_locked"],
        "degraded": bool(semantic and semantic["degraded"]),
        "dry_run": args.dry_run,
        "duration_ms": exact["duration_ms"] + (semantic["duration_ms"] if semantic else 0),
        "exact": _public(exact),
        "semantic": _public(semantic) if semantic else None,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"阈值 {report['threshold']}；回填 {report['backfilled']} 行；"
            f"精确归并 {report['exact_merged']} 行；语义归并 {report['semantic_merged']} 行；"
            f"簇 {report['clusters']} 个；降级 {'是' if report['degraded'] else '否'}"
        )
    if args.require_semantic and report["degraded"]:
        return 2
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(run(parse_args(sys.argv[1:]))))


if __name__ == "__main__":
    main()
