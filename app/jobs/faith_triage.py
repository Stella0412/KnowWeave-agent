"""编造个案台账的处置入口。

报告是产物，重跑一次就被覆盖；台账才是跨轮追溯的地方 —— 同一道题反复被判编造，
说明库里那一格一直没补对。这个 CLI 就是人工把个案点掉的按钮。

用法：
    uv run python -m app.jobs.faith_triage --list
    uv run python -m app.jobs.faith_triage --list --status 未解决
    uv run python -m app.jobs.faith_triage --resolve B03 --status 已解决 --resolution "补了 MH-CAM1 规格条目"
"""

import argparse
import asyncio
import sys

from app.db.engine import SessionFactory
from app.db.repository import FaithCaseRepository

_STATUS_CHOICES = ["未解决", "已解决", "无需解决"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="编造个案台账处置")
    parser.add_argument("--list", action="store_true", help="列出个案（默认动作）")
    parser.add_argument("--status", choices=_STATUS_CHOICES, default=None,
                        help="--list 时按状态筛选；--resolve 时是目标状态")
    parser.add_argument("--resolve", metavar="EVAL_ID", default=None, help="处置哪一道题")
    parser.add_argument("--resolution", default=None, help="处置说明（标已解决/无需解决时要写清）")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    if args.resolve and args.status is None:
        parser.error("--resolve 必须同时给 --status")
    return args


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def format_cases(rows: list) -> str:
    if not rows:
        return "台账暂无个案。"
    lines = []
    for row in rows:
        when = row.last_seen_at.strftime("%Y-%m-%d %H:%M") if row.last_seen_at else "-"
        lines.append(
            f"{row.eval_id} [{row.bucket}] 状态={_status_value(row.status)} "
            f"累计 {row.seen_count} 次 · 最近 {when}\n"
            f"    问题：{row.query}\n"
            f"    编造理由：{row.reason}\n"
            f"    处置说明：{row.resolution or '（未写）'}"
        )
    return "\n".join(lines)


async def _run(args: argparse.Namespace, session_factory) -> int:
    async with session_factory() as session:
        repo = FaithCaseRepository(session)
        if args.resolve:
            row = await repo.set_status(args.resolve, args.status, resolution=args.resolution)
            if row is None:
                print(f"台账里没有 eval_id={args.resolve} 的个案", file=sys.stderr)
                return 1
            await session.commit()
            print(f"已把 {args.resolve} 标为「{args.status}」")
            return 0
        rows = await repo.list_cases(status=args.status, limit=args.limit)
    print(format_cases(rows))
    return 0


async def run(args: argparse.Namespace, session_factory=None) -> int:
    return await _run(args, session_factory or SessionFactory)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(sys.argv[1:])
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
