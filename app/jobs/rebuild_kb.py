"""一次性的全量重建 CLI：drop Milvus 集合 + 清空 knowledge_chunks → 全量 build_kb。

为什么不能只用 `build_kb --rebuild`：它的增量判据是 `doc_root_exists(H1)`，而新夹具
`after-sales-manual.md` 的 H1「售后手册」与被删掉的旧文档**同名** → 增量路径会误判
「已入库」直接跳过，留下半新半旧的库，且不报错（spec §6.1）。

另外 Milvus 的 schema 变了（新增 BM25 函数与稀疏索引），而**函数与分词器只能在建集合时配置**
→ 集合必须 drop 重建。

用法：
    uv run python -m app.jobs.rebuild_kb --yes     # 停掉 uvicorn 之后跑
"""

import argparse
import asyncio
import sys

from sqlalchemy import delete

from app.db.engine import SessionFactory
from app.db.models import KnowledgeChunk
from app.jobs.build_kb import open_vector_store, run as run_build
from app.kb.milvus_client import MilvusKnowledgeClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清库重建：drop 集合 + 清空 chunk 表 + 全量建库")
    parser.add_argument("--docs-dir", default=None, help="知识文档目录（默认取 KB_DOCS_DIR）")
    parser.add_argument("--yes", action="store_true", help="确认已停掉 uvicorn（Milvus Lite 独占锁）")
    return parser.parse_args(argv)


async def reset() -> int:
    """清空权威源并 drop 派生读模型。返回清掉的 chunk 行数。"""
    async with SessionFactory() as session:
        result = await session.execute(delete(KnowledgeChunk))
        await session.commit()
        cleared = result.rowcount or 0

    store: MilvusKnowledgeClient = open_vector_store()
    try:
        store.drop_collection()
    finally:
        store.close()
    return cleared


async def _main(args: argparse.Namespace) -> int:
    """reset 与 build 必须在**同一个事件循环**里跑（见 main 的注释）。"""
    cleared = await reset()
    print(f"[重建] 已清空 knowledge_chunks {cleared} 行，Milvus 集合已 drop")
    # build_kb 自己会 ensure_collection（这次建出来的是 schema v2）与建表
    build_args = argparse.Namespace(docs_dir=args.docs_dir, rebuild=True)
    return await run_build(build_args)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(sys.argv[1:])
    if not args.yes:
        print("重建会清空 knowledge_chunks 并重建 Milvus 集合。")
        print("请先停掉 uvicorn（Milvus Lite 独占锁），再带上 --yes 重跑。")
        raise SystemExit(2)
    # 只能 asyncio.run 一次：`app.db.engine.engine` 是模块级、默认连接池，
    # 第一个循环借出的连接会带着绑定在已关闭循环上的 transport 回到池里，
    # 第二个 asyncio.run 复用它就会抛 `AttributeError: 'NoneType' object has no
    # attribute 'send'`（aiomysql）。两个阶段同循环即可。
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
