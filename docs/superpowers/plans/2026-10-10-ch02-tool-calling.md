name: ${MYSQL_COMPOSE_PROJECT:-knowweave}

# ch02 Function Calling 工具链 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给「知序」客服装上 Function Calling 工具链：模型自己决定调五个工具（订单/商品/物流 mock、FAQ LIKE 查表、建工单），单轮调用后回灌收敛，最终回答仍流式；聊天记录（含工具调用与结果）落四张 MySQL 表。

**Architecture:** FastAPI 分层（routers → services → tools → db/repository → core/prompts）。DB 是唯一真相源：每轮落 `messages(user)` → 读历史 + 内存 token 裁剪 → `bind_tools` 决策（`ainvoke`）→ 命中工具则推 `tool_status` 帧 + 执行（校验/超时/重试）→ 落 assistant(tool_calls) 与 tool 消息 → 回灌 → 最终回答 `astream` 流式 + 落 assistant 全文。`create_ticket` 把 conversation.status 置 `已转人工`。

**Tech Stack:** Python 3.14、FastAPI（内置 `fastapi.sse`）、SQLAlchemy 2.0 async + aiomysql、LangChain（`langchain_core.tools` 的 `@tool`）、Pydantic v2、tiktoken、uv；测试 pytest / pytest-asyncio / httpx + aiosqlite（内存库）。

**Spec:** `docs/superpowers/specs/2026-10-10-ch02-tool-calling-design.md`
**建表 DDL（列结构唯一真源）:** 仓库根 `建表语句.md`

## Global Constraints

- Python 3.14.7，依赖用 `uv` 管理（本机 uv 0.12.12）
- 建表列结构以仓库根 `建表语句.md` 为准，**不自行增删列**；ORM 字段名/枚举值逐字对齐 DDL
- 会话 status 枚举：`进行中`/`已转人工`/`已结束`（ch02 只触发 进行中→已转人工）；消息 role：`user`/`assistant`/`tool`；工单 ticket_type：`售后`/`投诉`/`咨询`；工单 status：`待处理`/`已处理`
- `session_id`（API 层）语义 = `str(conversations.id)`；首条消息 `session_id` 传 null → 后端 INSERT 会话拿数字 id → 经 `{"type":"session","session_id":"<数字id>"}` 回传；`user_id` 统一填 `"anonymous"`
- 商品/订单/物流三个工具内部随机造数、回显入参，不接真实接口、不建表
- 只做单轮：决策调用 `bind_tools`，最终回答调用**不绑工具**的裸模型（杜绝循环）
- SSE 用 FastAPI 内置 `fastapi.sse.EventSourceResponse`/`ServerSentEvent`，**不用 `sse-starlette`**
- 应用侧统一 OpenAI 协议：`ChatOpenAI(base_url/api_key/model)` 全从 `.env` 读；demo 模型 DeepSeek `deepseek-chat`
- 密钥只在 `.env`，不入库；缺 `LLM_API_KEY` 启动校验（ch01 已有）
- 聊天页改造走 Vibe Coding（不套 TDD/code review）；纯 Prompt/数据类任务用标注样例替代 TDD

## Review Focus

以下输入/失败模式是 spec 隐含但易漏测的，逐条对应到所属任务的测试：

1. **空 message** → `/chat` 返回 422，不落库、不调模型（Task 7）
2. **DB 不可用**（session_factory 打开失败）→ 产出 `{"type":"error"}` SSE 事件，不 500 崩（Task 7）
3. **query_faq 查不到** → 工具返回「未找到相关常见问题」，模型如实说明不编造（Task 4 单测 + Task 10 eval）
4. **工具抛错 / 超时 / 未知工具名** → executor 返回错误 ToolMessage，流不中断（Task 5）
5. **命中工具时四类消息落库齐全且顺序正确**：user → assistant(tool_calls) → tool → assistant(全文)（Task 7）

---

## Task 1: 依赖 + 配置 + Docker

**Files:**
- Modify: `pyproject.toml`、`app/config.py`、`.env.example`
- Create: `docker-compose.yml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces: `app.config.Settings` 新增字段 `mysql_host/mysql_port/mysql_user/mysql_password/mysql_db/tool_timeout/tool_retries`；`docker-compose.yml`（MySQL 8）

- [ ] **Step 1: 加依赖**

```bash
uv add "sqlalchemy[asyncio]" aiomysql
uv add --dev aiosqlite
```

`pyproject.toml` 应新增 `sqlalchemy>=2.0`（含 greenlet）、`aiomysql`，dev 组新增 `aiosqlite`。

- [ ] **Step 2: 写失败测试**

`tests/test_config.py` 追加：

```python
from app.config import Settings


def test_settings_mysql_defaults():
    s = Settings(_env_file=None)
    assert s.mysql_host == "127.0.0.1"
    assert s.mysql_port == 3306
    assert s.mysql_user == "root"
    assert s.mysql_db == "knowweave"
    assert s.tool_timeout == 5.0
    assert s.tool_retries == 1


def test_settings_mysql_from_env(monkeypatch):
    monkeypatch.setenv("MYSQL_HOST", "10.0.0.9")
    monkeypatch.setenv("MYSQL_PORT", "3307")
    monkeypatch.setenv("MYSQL_PASSWORD", "p@ss")
    monkeypatch.setenv("TOOL_TIMEOUT", "8")
    s = Settings(_env_file=None)
    assert s.mysql_host == "10.0.0.9"
    assert s.mysql_port == 3307
    assert s.mysql_password == "p@ss"
    assert s.tool_timeout == 8.0
```

- [ ] **Step 3: 运行测试确认失败**

```bash
uv run pytest tests/test_config.py -v
```

预期：FAIL（`AttributeError: 'Settings' object has no attribute 'mysql_host'` 或校验报错）

- [ ] **Step 4: 写实现**

`app/config.py` 增补字段（在 `token_budget` 后追加）：

```python
    # MySQL（Docker 起，连接参数从 .env 读）
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "knowweave123"
    mysql_db: str = "knowweave"
    # 工具执行
    tool_timeout: float = 5.0
    tool_retries: int = 1
```

`.env.example` 追加：

```bash
# MySQL（docker compose 起的库）
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=knowweave123
MYSQL_DB=knowweave
# 工具执行超时(秒)与重试次数
TOOL_TIMEOUT=5.0
TOOL_RETRIES=1
```

`docker-compose.yml`：

```yaml
services:
  mysql:
    image: mysql:8.0
    container_name: ${MYSQL_CONTAINER_NAME:-knowweave-mysql}
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_PASSWORD:-knowweave123}
      MYSQL_DATABASE: ${MYSQL_DB:-knowweave}
    ports:
      - "3306:3306"
    command: >-
      --default-authentication-plugin=mysql_native_password
      --character-set-server=utf8mb4
      --collation-server=utf8mb4_unicode_ci
    volumes:
      - knowweave_mysql_data:/var/lib/mysql
volumes:
  knowweave_mysql_data:
    name: ${MYSQL_VOLUME_NAME:-knowweave_mysql_data}
```

- [ ] **Step 5: 运行测试确认通过**

```bash
uv run pytest tests/test_config.py -v
```

预期：PASS（原有 2 条 + 新增 2 条）

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml uv.lock .env.example app/config.py tests/test_config.py docker-compose.yml
git commit -m "chore: MySQL 连接参数 + 工具超时重试配置 + docker-compose"
```

---

## Task 2: DB 基建（base / models / engine）

**Files:**
- Create: `app/db/__init__.py`、`app/db/base.py`、`app/db/models.py`、`app/db/engine.py`、`tests/conftest.py`、`tests/test_models.py`

**Interfaces:**
- Consumes: `app.config.Settings`、`build_db_url`（Task 1 产出的 `app.db.engine` 内）
- Produces: `app.db.base.Base`；`app.db.models.ConversationStatus/MessageRole/TicketType/TicketStatus`（str 枚举）、`Conversation/Message/FAQ/Ticket`；`app.db.engine.engine`（模块级 MySQL 引擎）、`app.db.engine.SessionFactory`（模块级 async_sessionmaker）、`app.db.engine.create_session_factory(url) -> async_sessionmaker`

- [ ] **Step 1: 写失败测试**

`tests/conftest.py`（供后续所有 DB 测试复用；SQLite 内存库 + StaticPool 共享单连接）：

```python
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()
```

`tests/test_models.py`：

```python
from app.db.base import Base
from app.db.models import Conversation, ConversationStatus, FAQ, Message, Ticket


def test_tables_registered():
    assert set(Base.metadata.tables) == {"conversations", "messages", "faq", "tickets"}


def test_conversation_columns():
    cols = {c.name for c in Conversation.__table__.columns}
    assert {"id", "user_id", "status", "created_at", "updated_at"} <= cols


def test_message_columns():
    cols = {c.name for c in Message.__table__.columns}
    assert {"id", "conversation_id", "role", "content", "tool_calls", "tool_call_id", "created_at"} <= cols


def test_ticket_columns():
    cols = {c.name for c in Ticket.__table__.columns}
    assert {"ticket_no", "conversation_id", "description", "ticket_type", "status", "created_at"} <= cols


def test_faq_columns():
    cols = {c.name for c in FAQ.__table__.columns}
    assert {"id", "question", "answer", "category", "created_at", "updated_at"} <= cols


def test_enum_values():
    assert {m.value for m in ConversationStatus} == {"进行中", "已转人工", "已结束"}
    assert {m.value for m in TicketType} == {"售后", "投诉", "咨询"}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_models.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.db'`）

- [ ] **Step 3: 写实现**

`app/db/base.py`：

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

`app/db/models.py`：

```python
import enum
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# SQLite 只对 INTEGER PRIMARY KEY 自增；BigInteger 主键在测试库需降级为 Integer
BIGINT = BigInteger().with_variant(Integer, "sqlite")


class ConversationStatus(str, enum.Enum):
    ACTIVE = "进行中"
    ESCALATED = "已转人工"
    CLOSED = "已结束"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TicketType(str, enum.Enum):
    AFTER_SALE = "售后"
    COMPLAINT = "投诉"
    CONSULT = "咨询"


class TicketStatus(str, enum.Enum):
    PENDING = "待处理"
    DONE = "已处理"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[ConversationStatus] = mapped_column(
        Enum(ConversationStatus), default=ConversationStatus.ACTIVE
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("conversations.id"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FAQ(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("conversations.id"), index=True
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[TicketType] = mapped_column(Enum(TicketType))
    status: Mapped[TicketStatus] = mapped_column(
        Enum(TicketStatus), default=TicketStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

`app/db/engine.py`：

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, settings


def build_db_url(s: Settings | None = None) -> str:
    s = s or settings
    return (
        f"mysql+aiomysql://{s.mysql_user}:{s.mysql_password}"
        f"@{s.mysql_host}:{s.mysql_port}/{s.mysql_db}?charset=utf8mb4"
    )


def create_session_factory(url: str) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(url)
    return async_sessionmaker(engine, expire_on_commit=False)


engine = create_async_engine(build_db_url(settings))
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)
```

`app/db/__init__.py`：空文件。

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_models.py -v
```

预期：PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add app/db/__init__.py app/db/base.py app/db/models.py app/db/engine.py tests/conftest.py tests/test_models.py
git commit -m "feat: SQLAlchemy 基建(四表 ORM + 异步 engine)"
```

---

## Task 3: Repository（四表读写 + ticket_no 生成）

**Files:**
- Create: `app/db/repository.py`
- Test: `tests/test_repository.py`

**Interfaces:**
- Consumes: `app.db.models.*`（Task 2）、`AsyncSession`
- Produces: `app.db.repository.ChatRepository(session)`，方法：`create_conversation(user_id="anonymous") -> int`、`get_conversation(conversation_id) -> Conversation | None`、`set_conversation_status(conversation_id, status) -> None`、`add_message(conversation_id, *, role, content=None, tool_calls=None, tool_call_id=None) -> None`、`list_messages(conversation_id) -> list[Message]`、`search_faq(keyword) -> list[FAQ]`、`create_ticket(conversation_id, description, ticket_type) -> str`（返回 ticket_no）

- [ ] **Step 1: 写失败测试**

`tests/test_repository.py`：

```python
import pytest

from app.db.models import ConversationStatus, TicketType
from app.db.repository import ChatRepository


@pytest.mark.asyncio
async def test_create_conversation_defaults(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await session.commit()
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.user_id == "anonymous"
    assert conv.status == ConversationStatus.ACTIVE


@pytest.mark.asyncio
async def test_add_and_list_messages_ordered(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        await repo.add_message(cid, role="user", content="你好")
        await repo.add_message(cid, role="assistant", content="您好")
        await session.commit()
        rows = await repo.list_messages(cid)
    assert [r.role.value for r in rows] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_search_faq_like(session_factory):
    from app.db.models import FAQ
    async with session_factory() as session:
        session.add(FAQ(question="退货政策是什么", answer="7天无理由退货", category="售后"))
        session.add(FAQ(question="发货时间多久", answer="48小时内发货", category="物流"))
        await session.commit()
        hits = await ChatRepository(session).search_faq("退货")
        miss = await ChatRepository(session).search_faq("邮费")
    assert len(hits) == 1
    assert hits[0].question == "退货政策是什么"
    assert miss == []


@pytest.mark.asyncio
async def test_create_ticket_generates_no_and_escapes(session_factory):
    async with session_factory() as session:
        repo = ChatRepository(session)
        cid = await repo.create_conversation()
        no = await repo.create_ticket(cid, "商品坏了", TicketType.AFTER_SALE)
        await repo.set_conversation_status(cid, ConversationStatus.ESCALATED)
        await session.commit()
    assert no.startswith("T") and len(no) == len("T") + 8 + 3
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.status == ConversationStatus.ESCALATED
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_repository.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.db.repository'`）

- [ ] **Step 3: 写实现**

`app/db/repository.py`：

```python
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    ConversationStatus,
    FAQ,
    Message,
    MessageRole,
    Ticket,
    TicketStatus,
    TicketType,
)


class ChatRepository:
    """会话/消息/工单/FAQ 的读写封装。方法只改 session，由调用方 commit。"""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create_conversation(self, user_id: str = "anonymous") -> int:
        conv = Conversation(user_id=user_id)
        self._session.add(conv)
        await self._session.flush()
        return conv.id

    async def get_conversation(self, conversation_id: int) -> Conversation | None:
        return await self._session.get(Conversation, conversation_id)

    async def set_conversation_status(
        self, conversation_id: int, status: ConversationStatus
    ) -> None:
        conv = await self.get_conversation(conversation_id)
        if conv is not None:
            conv.status = status

    async def add_message(
        self,
        conversation_id: int,
        *,
        role: MessageRole | str,
        content: str | None = None,
        tool_calls: list | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        self._session.add(
            Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )
        )

    async def list_messages(self, conversation_id: int) -> list[Message]:
        result = await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars())

    async def search_faq(self, keyword: str) -> list[FAQ]:
        like = f"%{keyword}%"
        result = await self._session.execute(
            select(FAQ).where(FAQ.question.like(like) | FAQ.answer.like(like))
        )
        return list(result.scalars())

    async def create_ticket(
        self, conversation_id: int, description: str, ticket_type: TicketType | str
    ) -> str:
        prefix = "T" + datetime.now().strftime("%Y%m%d")
        result = await self._session.execute(
            select(func.count())
            .select_from(Ticket)
            .where(Ticket.ticket_no.like(prefix + "%"))
        )
        seq = (result.scalar() or 0) + 1
        ticket_no = f"{prefix}{seq:03d}"
        self._session.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
                status=TicketStatus.PENDING,
            )
        )
        return ticket_no
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_repository.py -v
```

预期：PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add app/db/repository.py tests/test_repository.py
git commit -m "feat: ChatRepository 四表读写 + 工单号生成"
```

---

## Task 4: 工具定义 + 注册（5 个 @tool）

**Files:**
- Create: `app/tools/__init__.py`、`app/tools/business.py`、`app/tools/registry.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `ChatRepository`（Task 3）、`AsyncSession`/`async_sessionmaker`
- Produces: `app.tools.business.query_order/query_product/query_logistics`（模块级 @tool）、`build_query_faq(session_factory)`、`build_create_ticket(session_factory, conversation_id)`；`app.tools.registry.build_tools(session_factory, conversation_id) -> list[BaseTool]`（顺序：order/product/logistics/faq/create_ticket）

- [ ] **Step 1: 写失败测试**

`tests/test_tools.py`：

```python
import pytest

from app.db.repository import ChatRepository
from app.tools.business import build_create_ticket, build_query_faq, query_logistics, query_order
from app.tools.registry import build_tools


@pytest.mark.asyncio
async def test_mock_tools_echo_input():
    out = await query_order.ainvoke({"order_id": "1001"})
    assert "1001" in out
    out2 = await query_logistics.ainvoke({"order_id": "1001"})
    assert "1001" in out2


@pytest.mark.asyncio
async def test_query_faq_hit_and_miss(session_factory):
    from app.db.models import FAQ
    async with session_factory() as session:
        session.add(FAQ(question="退货政策是什么", answer="7天无理由退货", category="售后"))
        await session.commit()
    tool = build_query_faq(session_factory)
    hit = await tool.ainvoke({"keyword": "退货"})
    miss = await tool.ainvoke({"keyword": "邮费"})
    assert "7天无理由退货" in hit
    assert "未找到" in miss


@pytest.mark.asyncio
async def test_create_ticket_writes_and_escapes(session_factory):
    async with session_factory() as session:
        cid = await ChatRepository(session).create_conversation()
        await session.commit()
    tool = build_create_ticket(session_factory, cid)
    out = await tool.ainvoke({"description": "商品坏了", "ticket_type": "售后"})
    assert "工单" in out
    async with session_factory() as session:
        conv = await ChatRepository(session).get_conversation(cid)
    assert conv.status.value == "已转人工"


@pytest.mark.asyncio
async def test_build_tools_returns_five_named_tools(session_factory):
    tools = build_tools(session_factory, 1)
    assert [t.name for t in tools] == [
        "query_order", "query_product", "query_logistics", "query_faq", "create_ticket",
    ]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_tools.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.tools'`）

- [ ] **Step 3: 写实现**

`app/tools/business.py`：

```python
import random
from typing import Literal

from langchain_core.tools import BaseTool, tool

from app.db.models import ConversationStatus
from app.db.repository import ChatRepository


@tool
async def query_order(order_id: str) -> str:
    """查询订单信息。当用户询问订单状态、订单内容时调用。"""
    statuses = ["待付款", "已付款", "已发货", "已签收"]
    return (
        f"订单 {order_id}：状态={random.choice(statuses)}，"
        f"金额=¥{random.randint(20, 5000)}，商品=知序精选商品"
    )


@tool
async def query_product(product_name: str) -> str:
    """查询商品信息。当用户询问商品价格、库存、规格时调用。"""
    return (
        f"商品「{product_name}」：价格=¥{random.randint(9, 999)}，"
        f"库存={random.randint(0, 500)}件，规格=标准款"
    )


@tool
async def query_logistics(order_id: str) -> str:
    """查询物流信息。当用户询问物流到哪了、什么时候送达时调用。"""
    carriers = ["中通", "圆通", "顺丰", "韵达"]
    nodes = ["已揽收", "运输中", "到达转运中心", "派送中"]
    return (
        f"订单 {order_id} 物流：承运商={random.choice(carriers)}，"
        f"当前节点={random.choice(nodes)}，预计3日内送达"
    )


def build_query_faq(session_factory) -> BaseTool:
    @tool
    async def query_faq(keyword: str) -> str:
        """查询常见问题。当用户询问退货/换货/发货等政策类问题时调用。"""
        async with session_factory() as session:
            rows = await ChatRepository(session).search_faq(keyword)
        if not rows:
            return "未找到相关常见问题"
        return "\n".join(f"问：{r.question}\n答：{r.answer}" for r in rows)

    return query_faq


def build_create_ticket(session_factory, conversation_id: int) -> BaseTool:
    @tool
    async def create_ticket(
        description: str, ticket_type: Literal["售后", "投诉", "咨询"]
    ) -> str:
        """创建人工工单。当用户诉求需要人工介入处理时调用。"""
        async with session_factory() as session:
            repo = ChatRepository(session)
            ticket_no = await repo.create_ticket(conversation_id, description, ticket_type)
            await repo.set_conversation_status(conversation_id, ConversationStatus.ESCALATED)
            await session.commit()
        return f"已为您创建工单，工单号 {ticket_no}，人工客服会尽快处理。"

    return create_ticket
```

`app/tools/registry.py`：

```python
from langchain_core.tools import BaseTool

from app.tools.business import (
    build_create_ticket,
    build_query_faq,
    query_logistics,
    query_order,
    query_product,
)


def build_tools(session_factory, conversation_id: int) -> list[BaseTool]:
    return [
        query_order,
        query_product,
        query_logistics,
        build_query_faq(session_factory),
        build_create_ticket(session_factory, conversation_id),
    ]
```

`app/tools/__init__.py`：空文件。

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_tools.py -v
```

预期：PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add app/tools/__init__.py app/tools/business.py app/tools/registry.py tests/test_tools.py
git commit -m "feat: 五个业务工具(@tool) + 注册(依赖注入会话)"
```

---

## Task 5: 工具执行器（校验 + 超时重试 + 错误兜底）

**Files:**
- Create: `app/tools/executor.py`
- Test: `tests/test_executor.py`

**Interfaces:**
- Consumes: `langchain_core.tools.BaseTool`、`langchain_core.messages.ToolMessage`
- Produces: `app.tools.executor.execute_tool(tool_call: dict, tools_by_name: dict[str, BaseTool], *, timeout: float, retries: int) -> ToolMessage`

- [ ] **Step 1: 写失败测试**

`tests/test_executor.py`：

```python
import asyncio

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from app.tools.executor import execute_tool


@tool
def echo(x: str) -> str:
    """Echo."""
    return f"echo:{x}"


@tool
def boom(x: str) -> str:
    """Boom."""
    raise RuntimeError("kaboom")


@tool
async def slow(x: str) -> str:
    """Slow."""
    await asyncio.sleep(10)
    return "done"


def _call(name, args, timeout=5.0, retries=0):
    by_name = {"echo": echo, "boom": boom, "slow": slow}
    return execute_tool(
        {"name": name, "id": "call_1", "args": args},
        by_name,
        timeout=timeout,
        retries=retries,
    )


@pytest.mark.asyncio
async def test_valid_tool_returns_result():
    tm = await _call("echo", {"x": "hi"})
    assert isinstance(tm, ToolMessage)
    assert tm.content == "echo:hi"
    assert tm.tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_invalid_args_returns_error_message():
    tm = await _call("echo", {"wrong": "hi"})
    assert "校验失败" in tm.content


@pytest.mark.asyncio
async def test_tool_raises_returns_error_message():
    tm = await _call("boom", {"x": "hi"})
    assert "执行失败" in tm.content


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_message():
    tm = await execute_tool({"name": "nope", "id": "c", "args": {}}, {}, timeout=1.0, retries=0)
    assert "未知工具" in tm.content


@pytest.mark.asyncio
async def test_timeout_returns_error_message():
    tm = await _call("slow", {"x": "hi"}, timeout=0.05, retries=0)
    assert "执行失败" in tm.content
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_executor.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.tools.executor'`）

- [ ] **Step 3: 写实现**

`app/tools/executor.py`：

```python
import asyncio

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool


async def execute_tool(
    tool_call: dict,
    tools_by_name: dict[str, BaseTool],
    *,
    timeout: float,
    retries: int,
) -> ToolMessage:
    """执行单个工具调用：参数校验 → 超时重试 → 错误兜底，永远返回 ToolMessage 不抛。"""
    name = tool_call.get("name", "")
    call_id = tool_call.get("id", "")
    args = tool_call.get("args") or {}

    tool = tools_by_name.get(name)
    if tool is None:
        return ToolMessage(content=f"未知工具: {name}", tool_call_id=call_id, name=name)

    # 1. 参数 Schema 校验（args_schema 为 Pydantic 模型）
    if tool.args_schema is not None:
        try:
            tool.args_schema.model_validate(args)
        except Exception as e:
            return ToolMessage(
                content=f"工具参数校验失败: {e}", tool_call_id=call_id, name=name
            )

    # 2. 超时 + 重试执行
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
            content = result if isinstance(result, str) else str(result)
            return ToolMessage(content=content, tool_call_id=call_id, name=name)
        except Exception as e:  # 超时或工具自身报错都进兜底
            last_err = e

    return ToolMessage(
        content=f"工具执行失败: {last_err}", tool_call_id=call_id, name=name
    )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_executor.py -v
```

预期：PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add app/tools/executor.py tests/test_executor.py
git commit -m "feat: 工具执行器(参数校验 + 超时重试 + 错误兜底)"
```

---

## Task 6: 上下文裁剪重构（ContextStore 退化为纯裁剪器）

**Files:**
- Modify: `app/services/context_service.py`
- Test: Modify `tests/test_context_service.py`

**Interfaces:**
- Consumes: `tiktoken`、`langchain_core.messages.BaseMessage`
- Produces: `ContextStore(token_budget=4096)`，方法 `count_tokens(messages) -> int`、`trim(system_message, history) -> list[BaseMessage]`（纯函数，不再持会话状态）

- [ ] **Step 1: 改写失败测试**

`tests/test_context_service.py` 整体替换为：

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.context_service import ContextStore


def test_count_tokens_monotonic():
    store = ContextStore()
    short = store.count_tokens([HumanMessage(content="你好")])
    long = store.count_tokens([HumanMessage(content="你好" * 100)])
    assert short > 0
    assert long > short


def test_trim_drops_oldest_keeps_newest():
    store = ContextStore(token_budget=60)
    system = SystemMessage(content="你是客服")
    history = [AIMessage(content="第" + str(i) + "条很长的历史消息内容") for i in range(10)]
    history.append(HumanMessage(content="最新问题"))
    trimmed = store.trim(system, history)
    # 最新一条用户消息必须保留
    assert trimmed[-1].content == "最新问题"
    # system + trimmed 不超预算
    assert store.count_tokens([system] + trimmed) <= 60


def test_trim_does_not_mutate_input():
    store = ContextStore(token_budget=1000)
    history = [HumanMessage(content="你好")]
    store.trim(SystemMessage(content="sys"), history)
    assert len(history) == 1
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_context_service.py -v
```

预期：FAIL（`AttributeError: 'ContextStore' object has no attribute 'trim'`）

- [ ] **Step 3: 写实现**

`app/services/context_service.py` 整体替换为：

```python
import tiktoken
from langchain_core.messages import BaseMessage


class ContextStore:
    """纯 token 预算裁剪器：不再持有会话状态（DB 是唯一真相源），只做计数与裁剪。"""

    def __init__(self, token_budget: int = 4096):
        self._token_budget = token_budget
        self._enc = tiktoken.get_encoding("cl100k_base")  # 近似计数，DeepSeek 够用

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        total = 0
        for m in messages:
            total += 4  # 每条消息协议开销（近似）
            content = m.content if isinstance(m.content, str) else str(m.content)
            total += len(self._enc.encode(content))
        return total

    def trim(self, system_message: BaseMessage, history: list[BaseMessage]) -> list[BaseMessage]:
        """返回裁剪后的历史（不含 system），使 system+history 不超预算，且保留最新一条。"""
        history = list(history)  # 不修改调用方
        while (
            len(history) > 1
            and self.count_tokens([system_message] + history) > self._token_budget
        ):
            history.pop(0)
        return history
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_context_service.py -v
```

预期：PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add app/services/context_service.py tests/test_context_service.py
git commit -m "refactor: ContextStore 退化为纯 token 裁剪器(DB 已接管会话状态)"
```

---

## Task 7: 聊天编排改造（单轮工具 + 落库 + SSE tool_status）

**Files:**
- Modify: `app/services/chat_service.py`、`app/routers/chat.py`
- Test: Rewrite `tests/test_chat.py`

**Interfaces:**
- Consumes: `ChatRepository`（Task 3）、`build_tools`（Task 4）、`execute_tool`（Task 5）、`ContextStore`（Task 6）、`SessionFactory`（Task 2）、`build_chat_model`（ch01）
- Produces: `ChatService(model, session_factory, token_budget=4096, tool_timeout=5.0, tool_retries=1).stream(session_id, message) -> AsyncIterator[dict]`（yield `session`/`tool_status`/`token`/`done`/`error`）；`app.routers.chat.router`（懒实例化单例注入 `SessionFactory`）

- [ ] **Step 1: 写失败测试**

`tests/test_chat.py` 整体替换为：

```python
import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage

from app.db.models import Conversation
from app.routers import chat as chat_module
from app.services.chat_service import ChatService


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeBound:
    def __init__(self, message):
        self._message = message

    async def ainvoke(self, messages):
        return self._message


class FakeModel:
    def __init__(self, decision=None, chunks=()):
        self._decision = decision or AIMessage(content="直接回答")
        self._chunks = chunks or []

    def bind_tools(self, tools):
        return FakeBound(self._decision)

    def astream(self, messages):
        async def gen():
            for c in self._chunks:
                yield FakeChunk(c)
        return gen()


async def _collect(svc, session_id, message):
    events = []
    async for e in svc.stream(session_id, message):
        events.append(e)
    return events


@pytest.mark.asyncio
async def test_new_session_emits_session_and_persists(session_factory):
    svc = ChatService(FakeModel(), session_factory)
    events = await _collect(svc, None, "你好")
    assert events[0]["type"] == "session"
    assert events[-1]["type"] == "done"
    cid = int(events[0]["session_id"])
    async with session_factory() as session:
        conv = await session.get(Conversation, cid)
    assert conv is not None


@pytest.mark.asyncio
async def test_no_tool_path_streams_and_persists(session_factory):
    svc = ChatService(FakeModel(chunks=["您", "好"]), session_factory)
    events = await _collect(svc, None, "你好")
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == "您好"


@pytest.mark.asyncio
async def test_tool_path_full_roundtrip(session_factory):
    decision = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_logistics", "args": {"order_id": "1001"},
             "id": "call_1", "type": "tool_call"}
        ],
    )
    svc = ChatService(FakeModel(decision=decision, chunks=["您的", "包裹"]), session_factory)
    events = await _collect(svc, None, "订单1001到哪了")
    types = [e["type"] for e in events]
    assert "tool_status" in types
    ts = next(e for e in events if e["type"] == "tool_status")
    assert ts["name"] == "query_logistics"
    # 落库：user / assistant(tool_calls) / tool / assistant(全文) 四类齐全
    cid = int(events[0]["session_id"])
    async with session_factory() as session:
        from app.db.repository import ChatRepository
        rows = await ChatRepository(session).list_messages(cid)
    roles = [r.role.value for r in rows]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert rows[1].tool_calls[0]["name"] == "query_logistics"
    assert rows[2].tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_db_unavailable_emits_error(session_factory):
    class BoomFactory:
        def __call__(self):
            raise RuntimeError("db down")

    svc = ChatService(FakeModel(chunks=["x"]), BoomFactory())
    events = await _collect(svc, "1", "你好")
    assert events[-1]["type"] == "error"


async def test_chat_router_empty_message_422():
    app = FastAPI()
    app.include_router(chat_module.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/chat", json={"message": ""})
    assert r.status_code == 422
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_chat.py -v
```

预期：FAIL（ChatService 旧签名 `(model, store)` 与 `session_factory` 不匹配 → TypeError）

- [ ] **Step 3: 写实现**

`app/services/chat_service.py` 整体替换为：

```python
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.db.repository import ChatRepository
from app.prompts.customer_service import SYSTEM_PROMPT
from app.services.context_service import ContextStore
from app.tools.executor import execute_tool
from app.tools.registry import build_tools


def _row_to_message(row) -> BaseMessage:
    if row.role.value == "user":
        return HumanMessage(content=row.content or "")
    if row.role.value == "assistant":
        return AIMessage(content=row.content or "", tool_calls=row.tool_calls or [])
    return ToolMessage(content=row.content or "", tool_call_id=row.tool_call_id or "")


class ChatService:
    def __init__(
        self,
        model,
        session_factory,
        token_budget: int = 4096,
        tool_timeout: float = 5.0,
        tool_retries: int = 1,
    ):
        self._model = model
        self._session_factory = session_factory
        self._store = ContextStore(token_budget)
        self._tool_timeout = tool_timeout
        self._tool_retries = tool_retries

    async def stream(self, session_id: str | None, message: str):
        try:
            # 1. 解析/创建会话，落 user 消息，读历史
            async with self._session_factory() as session:
                repo = ChatRepository(session)
                if session_id is None:
                    conversation_id = await repo.create_conversation()
                    await session.commit()
                    yield {"type": "session", "session_id": str(conversation_id)}
                else:
                    conversation_id = int(session_id)
                await repo.add_message(conversation_id, role="user", content=message)
                await session.commit()
                rows = await repo.list_messages(conversation_id)
            history = [_row_to_message(r) for r in rows]

            # 2. 裁剪 + 决策
            system = SystemMessage(content=SYSTEM_PROMPT)
            trimmed = self._store.trim(system, history)
            decision_messages = [system] + trimmed
            tools = build_tools(self._session_factory, conversation_id)
            ai_msg = await self._model.bind_tools(tools).ainvoke(decision_messages)

            if ai_msg.tool_calls:
                # 3a. 命中工具：推状态帧 → 落 assistant(tool_calls) → 执行 → 落 tool → 回灌
                for tc in ai_msg.tool_calls:
                    yield {"type": "tool_status", "name": tc["name"], "status": "running"}
                async with self._session_factory() as session:
                    repo = ChatRepository(session)
                    await repo.add_message(
                        conversation_id, role="assistant", content="", tool_calls=ai_msg.tool_calls
                    )
                    await session.commit()
                by_name = {t.name: t for t in tools}
                tool_msgs = [
                    await execute_tool(
                        tc, by_name, timeout=self._tool_timeout, retries=self._tool_retries
                    )
                    for tc in ai_msg.tool_calls
                ]
                async with self._session_factory() as session:
                    repo = ChatRepository(session)
                    for tm in tool_msgs:
                        await repo.add_message(
                            conversation_id, role="tool", content=tm.content,
                            tool_call_id=tm.tool_call_id,
                        )
                    await session.commit()
                final_messages = decision_messages + [ai_msg] + tool_msgs
            else:
                # 3b. 未命中工具：直接回答
                final_messages = decision_messages

            # 4. 最终回答流式（不绑工具，杜绝循环）
            collected: list[str] = []
            async for chunk in self._model.astream(final_messages):
                if chunk.content:
                    collected.append(chunk.content)
                    yield {"type": "token", "content": chunk.content}

            # 5. 落 assistant 全文
            async with self._session_factory() as session:
                repo = ChatRepository(session)
                await repo.add_message(
                    conversation_id, role="assistant", content="".join(collected)
                )
                await session.commit()
            yield {"type": "done"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
```

`app/routers/chat.py` 整体替换为：

```python
from collections.abc import AsyncIterable

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.config import settings
from app.core.llm import build_chat_model
from app.db.engine import SessionFactory
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

# 懒实例化：避免 import 时用空 key 构造 ChatOpenAI 报错（ch01 Risk A）。
_chat_service: ChatService | None = None


def _get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(
            build_chat_model(settings),
            SessionFactory,
            token_budget=settings.token_budget,
            tool_timeout=settings.tool_timeout,
            tool_retries=settings.tool_retries,
        )
    return _chat_service


@router.post("", response_class=EventSourceResponse)
async def chat(req: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    async for event in _get_chat_service().stream(req.session_id, req.message):
        yield ServerSentEvent(data=event)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_chat.py -v
```

预期：PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add app/services/chat_service.py app/routers/chat.py tests/test_chat.py
git commit -m "feat: 聊天单轮工具编排(决策→执行→回灌→流式) + 落库 + SSE tool_status"
```

---

## Task 8: seed 灌数据 + 应用组装

**Files:**
- Create: `app/db/seed.py`
- Modify: `app/main.py`
- Test: `tests/test_seed.py`、Modify `tests/test_main.py`

**Interfaces:**
- Consumes: `Base`（Task 2）、`engine`（Task 2）、`ChatRepository`（Task 3）
- Produces: `app.db.seed.main()`（幂等建表 + 灌 faq）；`app.main.app` lifespan 内 `create_all`（幂等，DB 不可用仅告警不崩）

- [ ] **Step 1: 写失败测试**

`tests/test_seed.py`：

```python
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
```

`tests/test_main.py` 追加：

```python
def test_main_app_imports():
    import app.main  # noqa: F401  # 组装不抛即通过
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_seed.py tests/test_main.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.db.seed'`）

- [ ] **Step 3: 写实现**

`app/db/seed.py`：

```python
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
```

`app/main.py` 整体替换为：

```python
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import Settings, settings
from app.db.base import Base
from app.db.engine import engine
from app.routers import chat, extract

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def validate_config(s: Settings | None = None) -> None:
    s = s or settings
    if not s.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY，请在 .env 中配置后重启")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"[警告] 数据库不可用，跳过建表: {e}")
    yield


app = FastAPI(title="知序智能客服", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(extract.router)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_seed.py tests/test_main.py -v
```

预期：PASS（seed 1 passed；main 原有 2 + 新增 1）

- [ ] **Step 5: 提交**

```bash
git add app/db/seed.py app/main.py tests/test_seed.py tests/test_main.py
git commit -m "feat: seed 灌 faq 测试数据 + lifespan 幂等建表"
```

---

## Task 9: 前端工具轨迹徽章（Vibe Coding，直接改）

> 本任务走 Vibe Coding：直接改 `static/index.html`，不套 TDD/code review。验收靠浏览器手测。

**Files:**
- Modify: `static/index.html`

**目标效果：** ① 首条消息 `session_id` 传 null、由后端回传数字会话 id 后复用；② 收到 `tool_status` 事件时在客服气泡顶部挂「🔧 调用了 XX」小徽章，且收尾渲染 Markdown 时徽章不丢。

- [ ] **Step 1: 改 sessionId 生命周期**

`let sessionId = uuid();` → `let sessionId = null;`；`reset()` 内 `sessionId = uuid();` → `sessionId = null;`。（`uuid()` 函数可保留不动，`session` 事件已有 `sessionId = ev.session_id` 回写。）

- [ ] **Step 2: 加徽章样式**

在 `</style>` 前追加：

```css
  .tool-badge {
    display: inline-block;
    margin: 0 0 6px;
    padding: 3px 10px;
    border-radius: 12px;
    background: #fff1e6;
    color: #ff6f3c;
    border: 1px solid #ffd7bf;
    font-size: 12px;
  }
```

- [ ] **Step 3: handleLine 处理 tool_status**

在 `switch (ev.type)` 里 `case 'token':` 之前加：

```js
      case 'tool_status':
        if (!ctx.badge) {
          ctx.badge = document.createElement('div');
          ctx.badge.className = 'tool-badge';
          bubbleEl.insertBefore(ctx.badge, bubbleEl.firstChild);
        }
        ctx.badge.textContent = '🔧 调用了 ' + ev.name;
        break;
```

- [ ] **Step 4: finalizeAssistant 保留徽章**

`finalizeAssistant` 改为在重渲染时把徽章挪到顶部：

```js
  function finalizeAssistant(bubbleEl, ctx) {
    if (ctx.finished) return;
    ctx.finished = true;
    const badge = ctx.badge || bubbleEl.querySelector('.tool-badge');
    bubbleEl.innerHTML = '';
    if (badge) bubbleEl.appendChild(badge);
    const body = document.createElement('div');
    body.className = 'body';
    body.innerHTML = renderMarkdown(ctx.raw);
    bubbleEl.appendChild(body);
    scrollToBottom();
  }
```

- [ ] **Step 5: 提交**

```bash
git add static/index.html
git commit -m "feat: 聊天页工具轨迹徽章 + 会话 id 改后端下发"
```

---

## Task 10: 标注样例评估 + 端到端验收 + 留痕

**Files:**
- Create: `eval/samples_tools.jsonl`、`eval/eval_tools.py`、`dev-notes/ch02.md`

**Interfaces:**
- Consumes: `build_chat_model`（ch01）、`build_tools`（Task 4）、真实 DeepSeek
- Produces: 工具选择准确率（打印 + 记 dev-notes）；三条验收命令与结果

- [ ] **Step 1: 写标注样例**

`eval/samples_tools.jsonl`（每行 `{text, tool}`，`tool` 为期望选中的工具名）：

```json
{"text":"订单 1001 的物流到哪了","tool":"query_logistics"}
{"text":"订单 1001 现在什么状态","tool":"query_order"}
{"text":"这款猫粮多少钱","tool":"query_product"}
{"text":"退货政策是什么","tool":"query_faq"}
{"text":"商品坏了我要投诉，帮我转人工","tool":"create_ticket"}
{"text":"邮费是多少","tool":"query_faq"}
```

- [ ] **Step 2: 写评估脚本**

`eval/eval_tools.py`：

```python
import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.core.llm import build_chat_model
from app.db.engine import create_session_factory
from app.prompts.customer_service import SYSTEM_PROMPT
from app.tools.registry import build_tools


async def main():
    model = build_chat_model(settings)
    sf = create_session_factory("sqlite+aiosqlite:///:memory:")
    tools = build_tools(sf, 1)
    bound = model.bind_tools(tools)
    with open("eval/samples_tools.jsonl", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    ok = 0
    for s in samples:
        ai = await bound.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=s["text"])]
        )
        picked = ai.tool_calls[0]["name"] if ai.tool_calls else None
        hit = picked == s["tool"]
        ok += int(hit)
        print(f"[{s['text'][:14]}…] 期望={s['tool']} 实际={picked} {'✓' if hit else '✗'}")

    print(f"\n工具选择准确率: {ok}/{len(samples)}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: 跑工具选择评估并记录**

```bash
uv run python eval/eval_tools.py
```

记录准确率；其中「邮费是多少」期望选 query_faq 但查不到（漏召回），是预期行为，记入 dev-notes 留给 RAG 升级。

- [ ] **Step 4: 起 MySQL + 灌数据 + 启动服务**

```bash
docker compose up -d
uv run python -m app.db.seed
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- [ ] **Step 5: 三条验收（curl）**

```bash
# 验收①：订单物流 → 命中 query_logistics
curl -N -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"订单 1001 的物流到哪了"}'

# 验收②：退货政策 → query_faq 命中
curl -N -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"退货政策是什么"}'

# 验收③：邮费 → query_faq 查不到，如实说未查到（漏召回预期）
curl -N -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"邮费是多少"}'
```

> Windows 下中文 body 用 curl 内联会乱码，请用 UTF-8 文件 `--data-binary @file`。浏览器验收直接开 `http://127.0.0.1:8000/`，观察气泡徽章。

预期：① 出现 `tool_status`(query_logistics) 帧 + 按返回作答；② 命中 FAQ 作答；③ 工具被选中但查不到、如实说明。

- [ ] **Step 6: 提交 + 留痕**

```bash
git add eval/samples_tools.jsonl eval/eval_tools.py
git commit -m "test: 工具选择标注样例评估"
```

在 `dev-notes/ch02.md` 写过程留痕（brainstorm 定稿 / 计划评审 / 各任务完成 / code review 结论 / finish 三验收 + 评估结果 + 演示命令）。

---

## 收尾：全量测试

```bash
uv run pytest -q
```

预期：全部单测 PASS（ch01 存量 + ch02 新增）。三条验收 + 工具选择评估结果已在 dev-notes 记录。
