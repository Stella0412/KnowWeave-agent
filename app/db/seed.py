import asyncio
from datetime import datetime, timedelta

from sqlalchemy import func, select

from app.db.base import Base
from app.db.engine import engine
from app.db.models import FAQ, Conversation, Message, MessageRole, Order

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
    """给「对话挖知识」任务造可重复的演示会话。

    幂等判据 = 演示会话的**首句**是否已在本库里（不是「库里有没有会话」）：本仓库的
    MySQL 里通常已有 ch02 真实会话，按「有会话就返回」会让演示会话静默缺失，
    mine_qa 于是挖不到「发票 / 发货快递」这类只存在于演示会话里的问答。
    """
    first_turns = {turns[0][1] for turns in DEMO_CONVERSATIONS}
    async with session_factory() as session:
        existing = set(
            (
                await session.execute(
                    select(Message.content).where(Message.content.in_(first_turns))
                )
            ).scalars()
        )
        for turns in DEMO_CONVERSATIONS:
            if turns[0][1] in existing:
                continue
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


ORDER_SEED = [
    # order_no, product, type, amount, status, tracking, signed_days_ago
    ("1001", "猫爬架", "普通", 299, "已签收", "SF1001CN", 3),
    ("1002", "定制猫窝", "定制", 499, "已签收", "SF1002CN", 10),
    ("1003", "猫粮", "普通", 129, "已发货", "SF1003CN", None),
    ("1004", "猫砂", "普通", 59, "已取消", None, None),
    ("1005", "猫罐头", "生鲜", 89, "已签收", "SF1005CN", 2),
]


async def seed_orders(session_factory) -> None:
    async with session_factory() as session:
        count = (await session.execute(select(func.count()).select_from(Order))).scalar()
        if count:
            return
        now = datetime.now()
        for order_no, name, ptype, amount, status, tracking, signed_days in ORDER_SEED:
            signed_at = now - timedelta(days=signed_days) if signed_days is not None else None
            paid_at = now - timedelta(days=(signed_days or 1) + 1)
            session.add(
                Order(
                    order_no=order_no,
                    user_id="anonymous",
                    product_name=name,
                    product_type=ptype,
                    amount=amount,
                    status=status,
                    tracking_no=tracking,
                    paid_at=paid_at,
                    signed_at=signed_at,
                )
            )
        await session.commit()


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    from app.db.engine import SessionFactory

    await seed_faq(SessionFactory)
    await seed_conversations(SessionFactory)
    await seed_orders(SessionFactory)
    print("建表 + 灌入 faq 测试数据 + 演示会话 + 订单种子完成")


if __name__ == "__main__":
    asyncio.run(main())
