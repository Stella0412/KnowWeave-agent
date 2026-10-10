import asyncio

from sqlalchemy import func, select

from app.db.base import Base
from app.db.engine import engine
from app.db.models import FAQ

FAQ_ROWS = [
    ("退货政策是什么", "支持7天无理由退货，商品不影响二次销售即可申请。", "售后"),
    ("怎么申请退货", "在订单详情页点「申请售后」，按提示填写原因并寄回商品。", "售后"),
    ("换货怎么操作", "在订单详情页点「换货」，选择需要更换的规格即可。", "售后"),
    ("发货时间多久", "现货商品付款后48小时内发货，预售商品按页面标注时间。", "物流"),
    ("商品坏了怎么办", "联系人工客服或申请售后，我们会在24小时内核实处理。", "售后"),
]


async def seed_faq(session_factory) -> None:
    async with session_factory() as session:
        count = (await session.execute(select(func.count()).select_from(FAQ))).scalar()
        if count:
            return
        for question, answer, category in FAQ_ROWS:
            session.add(FAQ(question=question, answer=answer, category=category))
        await session.commit()


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    from app.db.engine import SessionFactory

    await seed_faq(SessionFactory)
    print("建表 + 灌入 faq 测试数据完成")


if __name__ == "__main__":
    asyncio.run(main())
