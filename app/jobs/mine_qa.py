"""离线挖知识 CLI：历史会话 → 暂存 → 去重 → 入 knowledge_chunks(pending) → 向量化补齐。

用法：
    uv run python -m app.jobs.mine_qa                     # 默认每批 20 个会话
    uv run python -m app.jobs.mine_qa --batch-size 5 --run-id mine-20261012
"""

import argparse
import asyncio
import sys

from app.config import settings
from app.core.embeddings import embed_texts
from app.core.llm import build_structured_model
from app.db.base import Base
from app.db.engine import SessionFactory, engine
from app.jobs.build_kb import open_vector_store
from app.kb.dualwrite import vectorize_pending
from app.kb.mining import QABatch, QAMiner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线挖知识：历史会话 → 暂存 → 去重 → 知识库")
    parser.add_argument("--batch-size", type=int, default=None, help="每批喂给 LLM 的会话数")
    parser.add_argument("--run-id", default=None, help="批次号前缀，同日重跑传相同值即可幂等")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    store = open_vector_store()
    try:
        model = build_structured_model(settings, QABatch)
        miner = QAMiner(
            SessionFactory,
            model,
            batch_size=args.batch_size or settings.mine_batch_size,
        )
        report = await miner.mine(run_id=args.run_id)
        print(
            f"[抽取] 会话 {report.conversations} 个 / {report.batches} 批，"
            f"抽出 {report.extracted} 条，保留 {report.kept}，去重丢弃 {report.discarded}"
        )
        n = await vectorize_pending(
            SessionFactory, embed=embed_texts, milvus=store, batch_size=200
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
