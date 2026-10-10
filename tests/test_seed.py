import pytest

from app.db.repository import ChatRepository
from app.db.seed import seed_faq


@pytest.mark.asyncio
async def test_seed_faq_idempotent(session_factory):
    await seed_faq(session_factory)
    await seed_faq(session_factory)  # 重复执行不重复插入
    async with session_factory() as session:
        rows = await ChatRepository(session).search_faq("退货")
        postage = await ChatRepository(session).search_faq("邮费")
    assert len(rows) >= 1
    assert postage == []  # 刻意不含邮费/运费，为验收③保留漏召回
