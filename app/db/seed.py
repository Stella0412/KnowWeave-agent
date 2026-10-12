import asyncio

from sqlalchemy import func, select

from app.db.base import Base
from app.db.engine import engine
from app.db.models import FAQ, Conversation, Message, MessageRole

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


DEMO_CONVERSATIONS = [
    [
        ("user", "你们发什么快递呀"),
        ("assistant", "默认中通快递，偏远地区可能转其他快递。"),
        ("user", "那能开发票吗"),
        ("assistant", "可以开发票，下单时备注抬头和税号即可，电子发票 3 个工作日内发到邮箱。"),
    ],
    [
        ("user", "我想问问能不能货到付款"),
        ("assistant", "目前支持货到付款，仅限部分城市，下单时选择「货到付款」即可看到是否支持。"),
        ("user", "包装严实吗，猫粮容易受潮"),
        ("assistant", "猫粮会用铝箔袋加纸箱双层包装，并在箱内加干燥剂。"),
    ],
]


async def seed_conversations(session_factory) -> None:
    """给「对话挖知识」任务造可重复的演示会话（幂等：已有会话则直接返回）。"""
    async with session_factory() as session:
        count = (await session.execute(select(func.count()).select_from(Conversation))).scalar()
        if count:
            return
        for turns in DEMO_CONVERSATIONS:
            conv = Conversation(user_id="anonymous")
            session.add(conv)
            await session.flush()
            for role, content in turns:
                session.add(
                    Message(
                        conversation_id=conv.id,
                        role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
                        content=content,
                    )
                )
        await session.commit()


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    from app.db.engine import SessionFactory

    await seed_faq(SessionFactory)
    await seed_conversations(SessionFactory)
    print("建表 + 灌入 faq 测试数据 + 演示会话完成")


if __name__ == "__main__":
    asyncio.run(main())
