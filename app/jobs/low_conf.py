"""低置信度问题池的查看入口。

池子只写不读就等于没有：验收④ 要求「问题进了低置信度池」得**当场看得见**。
归并与数据飞轮消费是 ch09 的事，本章只给这一个读口。

用法：
    uv run python -m app.jobs.low_conf --list
    uv run python -m app.jobs.low_conf --list --source self_check --limit 5
    uv run python -m app.jobs.low_conf --count
"""

import argparse
import asyncio
import sys

from app.db.engine import SessionFactory
from app.db.repository import LowConfidenceRepository


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="低置信度问题池查看")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="列出池中问题（默认动作）")
    group.add_argument("--count", action="store_true", help="只按 source 计数")
    parser.add_argument("--source", default=None, help="只看某个入池入口")
    parser.add_argument("--limit", type=int, default=20, help="列出条数上限")
    return parser.parse_args(argv)


def _source_value(source) -> str:
    return source.value if hasattr(source, "value") else str(source)


def format_questions(rows: list) -> str:
    if not rows:
        return "池中暂无疑问。"
    lines = []
    for row in rows:
        when = row.created_at.strftime("%Y-%m-%d %H:%M") if row.created_at else "-"
        lines.append(
            f"#{row.id} [{when}] ({_source_value(row.source)}) {row.raw_question}\n"
            f"    原因：{row.reason or '（未记录）'}"
        )
    return "\n".join(lines)


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "池中暂无疑问。"
    return "\n".join(f"{name}: {value}" for name, value in sorted(counts.items()))


async def run(args: argparse.Namespace) -> int:
    async with SessionFactory() as session:
        repo = LowConfidenceRepository(session)
        if args.count:
            print(format_counts(await repo.count_by_source()))
            return 0
        rows = await repo.list_questions(source=args.source, limit=args.limit)
    print(format_questions(rows))
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(run(parse_args(sys.argv[1:]))))


if __name__ == "__main__":
    main()
