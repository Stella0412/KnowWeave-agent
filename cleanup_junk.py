"""一次性清理：把「客服自己的非答复」被挖进来的 mined 条目从 MySQL + Milvus 里去掉。

只动 content_type == 'mined' 的行 —— 源文档块（manual/policy/faq）里出现的「转人工」
是政策正文，不是非答复，不能被闸门误删。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import select

from app.db.engine import SessionFactory
from app.db.models import KnowledgeChunk, StagingStatus
from app.db.repository import KnowledgeRepository
from app.kb.mining import is_non_answer
from app.kb.milvus_client import build_milvus_client


async def main() -> None:
    store = build_milvus_client()
    try:
        async with SessionFactory() as s:
            chunks = (await s.execute(select(KnowledgeChunk))).scalars().all()
            bad = [c for c in chunks if c.content_type == "mined" and is_non_answer(c.answer)]
            print(f"清理前：chunk={len(chunks)}  Milvus={store.count()}")
            for c in bad:
                print(f"  删除 id={c.id} q={c.questions[:18]!r} a={c.answer[:44]!r}")
            ids = [c.id for c in bad]
            await asyncio.to_thread(store.delete_ids, ids)
            for c in bad:
                await s.delete(c)
            repo = KnowledgeRepository(s)
            kept = await repo.list_staging(None, StagingStatus.KEPT)
            bad_kept = [r.id for r in kept if is_non_answer(r.answer)]
            await repo.set_staging_status(bad_kept, StagingStatus.DISCARDED)
            await s.commit()
            print(f"  暂存 KEPT 里另标 DISCARDED：{bad_kept}")
            left = len((await s.execute(select(KnowledgeChunk))).scalars().all())
        print(f"清理后：chunk={left}  Milvus={store.count()}")
    finally:
        store.close()


asyncio.run(main())
