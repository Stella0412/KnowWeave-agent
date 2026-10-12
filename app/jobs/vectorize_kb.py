"""只做阶段2：pending → Milvus → done（幂等可重跑）。

用法：
    uv run python -m app.jobs.vectorize_kb
    uv run python -m app.jobs.vectorize_kb --batch-size 100
"""

import argparse
import asyncio
import sys

from app.core.embeddings import embed_texts
from app.db.base import Base
from app.db.engine import SessionFactory, engine
from app.jobs.build_kb import open_vector_store
from app.kb.dualwrite import vectorize_pending


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="向量化补齐：pending → Milvus → done")
    parser.add_argument("--batch-size", type=int, default=200, help="每批处理条数")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    store = open_vector_store()
    try:
        n = await vectorize_pending(
            SessionFactory, embed=embed_texts, milvus=store, batch_size=args.batch_size
        )
        print(f"[阶段2] 向量化补齐 {n} 条")
    finally:
        store.close()
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(sys.argv[1:])
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
