"""离线建库 CLI：知识文档 → chunks → MySQL pending → 向量化补齐。

用法：
    uv run python -m app.jobs.build_kb              # 增量：跳过已入库文档，只补 pending 向量
    uv run python -m app.jobs.build_kb --rebuild    # 重切：先按文档根删除 MySQL 行 + Milvus 主键
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.config import settings
from app.core.embeddings import build_embeddings, embed_texts
from app.db.base import Base
from app.db.engine import SessionFactory, engine
from app.db.repository import KnowledgeRepository
from app.kb.chunking import ChunkLimits, first_heading
from app.kb.documents import build_chunks
from app.kb.dualwrite import vectorize_pending, write_chunks_pending
from app.kb.milvus_client import build_milvus_client
from app.kb.sources import list_sources

logger = logging.getLogger(__name__)

_LOCK_HINT = (
    "Milvus Lite 数据目录被占用：请先停掉正在运行的服务（uvicorn），"
    "或把 .env 的 MILVUS_URI 切到 http://127.0.0.1:19530 起 standalone 后重试。"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线建库：知识文档 → MySQL + Milvus 双写")
    parser.add_argument("--docs-dir", default=None, help="知识文档目录（默认取 KB_DOCS_DIR）")
    parser.add_argument("--rebuild", action="store_true", help="按文档根删除后重切")
    return parser.parse_args(argv)


def open_vector_store():
    """打开 Milvus。注意：Milvus Lite 的独占锁异常会被 server_manager 吞掉、最终以
    「连不上」的形式抛出，所以按「Lite 模式失败」给提示，同时打印原始错误。"""
    try:
        store = build_milvus_client(settings)
        store.ensure_collection()
        return store
    except Exception as exc:
        print(f"[建库失败] 打开 Milvus 出错: {exc}")
        if not settings.milvus_uri.startswith(("http://", "https://")):
            print(_LOCK_HINT)
        raise SystemExit(1)


async def run(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir or settings.kb_docs_dir)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    store = open_vector_store()
    limits = ChunkLimits(
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        table_max_rows=settings.chunk_table_max_rows,
    )
    try:
        processed = skipped = inserted = 0
        for source in list_sources(docs_dir):
            text = source.path.read_text(encoding="utf-8")
            root = first_heading(text)
            if root is None:
                logger.warning("跳过无标题文档（切分产出为 0）: %s", source.path)
                continue
            async with SessionFactory() as session:
                repo = KnowledgeRepository(session)
                if args.rebuild:
                    old_ids = await repo.delete_by_doc_root(root)
                    await session.commit()
                    if old_ids:
                        await asyncio.to_thread(store.delete_ids, old_ids)
                elif await repo.doc_root_exists(root):
                    skipped += 1
                    continue
                chunks = build_chunks(text, content_type=source.content_type, limits=limits)
                if not chunks:
                    logger.warning("切分产出 0 个 chunk，跳过: %s", source.path)
                    continue
                ids = await repo.insert_chunks(chunks)
                await repo.link_neighbors(ids)
                await session.commit()
            processed += 1
            inserted += len(ids)
        print(f"[阶段1] 文档 {processed} 篇入库（跳过 {skipped} 篇），新增 chunk {inserted} 条")

        vectorized = await vectorize_pending(
            SessionFactory, embed=embed_texts, milvus=store,
            batch_size=200,
        )
        print(f"[阶段2] 向量化补齐 {vectorized} 条")
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
