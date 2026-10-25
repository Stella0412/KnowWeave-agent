"""飞轮审核 CLI：列表/详情/补答/驳回/重开/指标。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.db.engine import SessionFactory
from app.services.flywheel_service import FlywheelService


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="低置信度问题池审核")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--list", action="store_true", help="列出待处理根问题")
    actions.add_argument("--show", type=int, metavar="ID", help="查看详情")
    actions.add_argument("--answer", type=int, metavar="ID", help="人工补答")
    actions.add_argument("--ignore", type=int, metavar="ID", help="驳回")
    actions.add_argument("--reopen", type=int, metavar="ID", help="重开")
    actions.add_argument("--metrics", action="store_true", help="池指标")
    parser.add_argument("--status", default="待处理")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--category", default="人工补充")
    parser.add_argument("--question-file")
    parser.add_argument("--answer-file")
    parser.add_argument("--resolution")
    args = parser.parse_args(argv)
    if args.answer is not None and (not args.question_file or not args.answer_file):
        parser.error("--answer 需要 --question-file 与 --answer-file")
    return args


async def run(args: argparse.Namespace) -> int:
    service = FlywheelService(SessionFactory)
    if args.metrics:
        print(json.dumps(await service.metrics(), ensure_ascii=False, indent=2))
        return 0
    if args.show is not None:
        print(json.dumps(await service.cluster_detail(args.show), ensure_ascii=False, indent=2))
        return 0
    if args.answer is not None:
        question = Path(args.question_file).read_text(encoding="utf-8").strip()
        answer = Path(args.answer_file).read_text(encoding="utf-8").strip()
        result = await service.submit_answer(
            args.answer, question=question, answer=answer, category=args.category,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.ignore is not None:
        if not args.resolution:
            raise ValueError("--ignore 需要 --resolution")
        print(json.dumps(
            await service.ignore(args.ignore, args.resolution),
            ensure_ascii=False, indent=2,
        ))
        return 0
    if args.reopen is not None:
        if not args.resolution:
            raise ValueError("--reopen 需要 --resolution")
        print(json.dumps(
            await service.reopen(args.reopen, args.resolution),
            ensure_ascii=False, indent=2,
        ))
        return 0
    items, total = await service.list_queue(
        status=args.status, limit=args.limit, offset=args.offset,
    )
    for item in items:
        print(
            f"#{item['id']} [{item['status']}] x{item['duplicate_count']} "
            f"{item['raw_question']}"
        )
    print(f"共 {total} 个根问题")
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(sys.argv[1:])
    try:
        raise SystemExit(asyncio.run(run(args)))
    except Exception as exc:  # noqa: BLE001
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
