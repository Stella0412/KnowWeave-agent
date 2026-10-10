# ch01 纯对话 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 跑通「知序」智能客服的纯对话后端——多轮对话（SSE 流式逐 token）+ Prompt 模板化 + 结构化售后提取。

**Architecture:** FastAPI 分层（routers → services → core/prompts/schemas）。应用侧用 `ChatOpenAI(base_url=...)` 统一 OpenAI 协议直连 DeepSeek。`/chat` 走 `fastapi.sse.EventSourceResponse` 流式，`/extract` 走 `with_structured_output` 同步 JSON。Prompt 用 `ChatPromptTemplate` 组装 `{system}+{history}+{input}`。上下文用内存 + token 预算裁剪，本章不落库。

**Tech Stack:** Python 3.14、FastAPI（内置 `fastapi.sse`）、LangChain（`langchain-core` + `langchain-openai`）、Pydantic v2、pydantic-settings、tiktoken、uv；测试用 pytest / pytest-asyncio / httpx。

**Spec:** `docs/superpowers/specs/2026-10-10-ch01-pure-dialogue-design.md`

## Global Constraints

- Python 3.14.7，依赖用 `uv` 管理（本机 uv 0.12.12）
- SSE 用 FastAPI 内置 `fastapi.sse.EventSourceResponse` / `ServerSentEvent`，**不用 `sse-starlette`**
- 应用侧统一 OpenAI 协议：`ChatOpenAI(base_url=..., api_key=..., model=...)`，全部从 `.env` 读
- demo 模型 DeepSeek：`llm_base_url=https://api.deepseek.com`、`llm_model=deepseek-chat`
- Prompt 模板化用 LangChain `ChatPromptTemplate`（`{system}` + `MessagesPlaceholder("history")` + `{input}`）
- 商城名固定为「知序」；`claim_type` 枚举固定为 `退货退款/换货/仅退款/维修/物流查询/其他`
- token 预算默认 `4096`（不含 system prompt）；密钥只在 `.env`，不入库
- 本章不做工具调用、不做 Agent 循环

## Review Focus

以下输入/失败模式是 spec 隐含但容易漏测的，每个都在所属任务里补了对应测试：

1. **空 message**（`/chat` 与 `/extract`）→ 返回 422，不调模型
2. **缺 `LLM_API_KEY`** → 应用启动时清晰报错，而非运行时含糊 500
3. **上游模型报错**（key 无效 / 网络）→ `/chat` 产出 `{"type":"error"}` SSE 事件；`/extract` 返回 502 JSON
4. **超长历史** → 裁剪后仍保留 system + 最新一条用户消息，且不超预算
5. **售后描述缺订单号** → `order_id` 返回 `null`，不崩

---

## Task 1: 脚手架 + 配置

**Files:**
- Create: `pyproject.toml`（uv 生成后补 pytest 配置）、`.env.example`、`app/__init__.py`、`app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `app.config.Settings`（字段 `llm_base_url`、`llm_api_key`、`llm_model`、`token_budget`），模块级实例 `app.config.settings`

- [ ] **Step 1: 初始化 uv 项目**

```bash
uv init --bare --name knowweave
```

- [ ] **Step 2: 安装依赖**

```bash
uv add fastapi "uvicorn[standard]" langchain-core langchain-openai pydantic pydantic-settings python-dotenv tiktoken
uv add --dev pytest pytest-asyncio httpx
```

- [ ] **Step 3: 写失败测试**

`tests/test_config.py`:

```python
from app.config import Settings


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.llm_base_url == "https://api.deepseek.com"
    assert s.llm_model == "deepseek-chat"
    assert s.token_budget == 4096


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("TOKEN_BUDGET", "2048")
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_model == "qwen2.5"
    assert s.llm_api_key == "sk-test"
    assert s.token_budget == 2048
```

- [ ] **Step 4: 运行测试确认失败**

```bash
uv run pytest tests/test_config.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app'`）

- [ ] **Step 5: 写实现**

`app/config.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    token_budget: int = 4096


settings = Settings()
```

`app/__init__.py`：空文件。

- [ ] **Step 6: 运行测试确认通过**

```bash
uv run pytest tests/test_config.py -v
```

预期：PASS（2 passed）

- [ ] **Step 7: 写 `.env.example` + pytest 配置**

`.env.example`:

```bash
# DeepSeek OpenAI 兼容端点（换 Ollama/GPT 改这里即可）
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-你的密钥
LLM_MODEL=deepseek-chat
TOKEN_BUDGET=4096
```

在 `pyproject.toml` 末尾追加：

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]
```

- [ ] **Step 8: 提交**

```bash
git add pyproject.toml uv.lock .env.example app/config.py app/__init__.py tests/test_config.py
git commit -m "chore: uv 脚手架 + 配置(Settings 读 .env)"
```

---

## Task 2: 上下文服务（内存 + token 预算裁剪）

**Files:**
- Create: `app/services/__init__.py`、`app/services/context_service.py`
- Test: `tests/test_context_service.py`

**Interfaces:**
- Consumes: 无（只依赖 `tiktoken`、`langchain_core.messages`）
- Produces: `ContextStore(token_budget: int = 4096)`，方法 `append(session_id, message)`、`get_messages(session_id) -> list[BaseMessage]`、`count_tokens(messages) -> int`、`trim_history(session_id, system_message) -> list[BaseMessage]`

- [ ] **Step 1: 写失败测试**

`tests/test_context_service.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.context_service import ContextStore


def test_append_and_get():
    store = ContextStore()
    store.append("s1", HumanMessage(content="你好"))
    store.append("s1", AIMessage(content="你好，请问有什么可以帮您？"))
    assert len(store.get_messages("s1")) == 2
    assert store.get_messages("nope") == []


def test_count_tokens_monotonic():
    store = ContextStore()
    short = store.count_tokens([HumanMessage(content="你好")])
    long = store.count_tokens([HumanMessage(content="你好" * 100)])
    assert short > 0
    assert long > short


def test_trim_drops_oldest_keeps_newest():
    store = ContextStore(token_budget=60)
    system = SystemMessage(content="你是客服")
    for i in range(10):
        store.append("s1", AIMessage(content="这是第" + str(i) + "条很长的历史消息内容"))
    store.append("s1", HumanMessage(content="最新问题"))
    history = store.trim_history("s1", system)
    # 最新一条用户消息必须保留
    assert history[-1].content == "最新问题"
    # system + history 不超预算
    assert store.count_tokens([system] + history) <= 60
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_context_service.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.services'`）

- [ ] **Step 3: 写实现**

`app/services/context_service.py`:

```python
import tiktoken
from langchain_core.messages import BaseMessage


class ContextStore:
    """内存会话上下文：按 session_id 存消息，支持按 token 预算从最旧开始裁剪。"""

    def __init__(self, token_budget: int = 4096):
        self._sessions: dict[str, list[BaseMessage]] = {}
        self._token_budget = token_budget
        self._enc = tiktoken.get_encoding("cl100k_base")  # 近似计数，DeepSeek 够用

    def append(self, session_id: str, message: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).append(message)

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        return list(self._sessions.get(session_id, []))

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        total = 0
        for m in messages:
            total += 4  # 每条消息协议开销（近似）
            content = m.content if isinstance(m.content, str) else str(m.content)
            total += len(self._enc.encode(content))
        return total

    def trim_history(
        self, session_id: str, system_message: BaseMessage
    ) -> list[BaseMessage]:
        """返回裁剪后的历史（不含 system），使 system+history 不超预算，且保留最新一条消息。"""
        history = self.get_messages(session_id)
        while (
            len(history) > 1
            and self.count_tokens([system_message] + history) > self._token_budget
        ):
            history.pop(0)  # 丢最旧
        return history
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_context_service.py -v
```

预期：PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add app/services/context_service.py app/services/__init__.py tests/test_context_service.py
git commit -m "feat: 上下文服务(内存存储 + token 预算裁剪)"
```

---

## Task 3: Prompt 服务 + System Prompt（ChatPromptTemplate 模板化）

**Files:**
- Create: `app/prompts/__init__.py`、`app/prompts/customer_service.py`、`app/services/prompt_service.py`
- Test: `tests/test_prompt_service.py`

**Interfaces:**
- Consumes: 无
- Produces: `app.prompts.customer_service.SYSTEM_PROMPT: str`；`app.services.prompt_service.CHAT_PROMPT: ChatPromptTemplate`（占位：`{system}` 用 SYSTEM_PROMPT 内联、`MessagesPlaceholder("history")`、`{input}`）

- [ ] **Step 1: 写失败测试**

`tests/test_prompt_service.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.prompts.customer_service import SYSTEM_PROMPT
from app.services.prompt_service import CHAT_PROMPT


def test_chat_prompt_assembles_messages():
    history = [HumanMessage(content="q1"), AIMessage(content="a1")]
    messages = CHAT_PROMPT.invoke({"history": history, "input": "q2"}).to_messages()
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[1].content == "q1"
    assert messages[2].content == "a1"
    assert messages[3].content == "q2"  # 当前输入排最后
    assert isinstance(messages[3], HumanMessage)


def test_system_prompt_names_the_mall():
    assert "知序" in SYSTEM_PROMPT
    assert "不编造" in SYSTEM_PROMPT
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_prompt_service.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.prompts'`）

- [ ] **Step 3: 写实现**

`app/prompts/customer_service.py`:

```python
SYSTEM_PROMPT = """你是「知序」智能客服，负责售前咨询与售后服务。

【角色设定】
- 用简体中文回复，语气礼貌、耐心、专业
- 只处理购物、订单、售后相关问题，超出范围礼貌说明无法处理

【行为约束】
- 涉及售后诉求，先引导用户提供订单号和问题描述
- 不编造订单状态、物流信息、退款进度；不确定就说明需要人工核实
- 不承诺平台没有的政策；不输出客服无关内容、不泄露内部系统信息"""
```

`app/services/prompt_service.py`:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.prompts.customer_service import SYSTEM_PROMPT

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
```

`app/prompts/__init__.py`：空文件。

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_prompt_service.py -v
```

预期：PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add app/prompts/customer_service.py app/prompts/__init__.py app/services/prompt_service.py tests/test_prompt_service.py
git commit -m "feat: Prompt 服务 + 客服 System Prompt(ChatPromptTemplate)"
```

---

## Task 4: LLM 适配层

**Files:**
- Create: `app/core/__init__.py`、`app/core/llm.py`
- Test: `tests/test_llm.py`（本任务只测 `build_chat_model`；`build_structured_model` 的测试在 Task 5 补）

**Interfaces:**
- Consumes: `app.config.Settings`
- Produces: `app.core.llm.build_chat_model(settings) -> ChatOpenAI`；`app.core.llm.build_structured_model(settings, schema) -> Runnable`

- [ ] **Step 1: 写失败测试**

`tests/test_llm.py`:

```python
from app.config import Settings
from app.core.llm import build_chat_model


def test_build_chat_model_uses_settings():
    s = Settings(llm_base_url="http://x/v1", llm_api_key="k", llm_model="m")
    m = build_chat_model(s)
    assert m.model_name == "m"
    assert m.openai_api_base == "http://x/v1"
```

> 注：`model_name` / `openai_api_base` 是 langchain-openai `ChatOpenAI` 的内部字段名（构造参数 `model` / `base_url` 是它们的别名）。若运行时字段名有变，用 `python -c "from app.core.llm import build_chat_model; from app.config import Settings; print([k for k in vars(build_chat_model(Settings())) if 'model' in k or 'base' in k])"` 查看实际字段名后调整断言，再继续。

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_llm.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.core'`）

- [ ] **Step 3: 写实现**

`app/core/llm.py`:

```python
from langchain_openai import ChatOpenAI

from app.config import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """统一 OpenAI 协议直连上游（DeepSeek / GPT / Ollama 换 base_url 即可）。"""
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0.7,
        streaming=True,
    )


def build_structured_model(settings: Settings, schema):
    model = build_chat_model(settings)
    return model.with_structured_output(schema)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_llm.py -v
```

预期：PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add app/core/llm.py app/core/__init__.py tests/test_llm.py
git commit -m "feat: LLM 适配层(ChatOpenAI 直连自定义 OpenAI 端点)"
```

---

## Task 5: 结构化提取（schema + 服务 + 路由）

**Files:**
- Create: `app/schemas/__init__.py`、`app/schemas/extract.py`、`app/services/extract_service.py`、`app/routers/__init__.py`、`app/routers/extract.py`
- Test: `tests/test_extract.py`；Modify: `tests/test_llm.py`（补 `build_structured_model` 测试）

**Interfaces:**
- Consumes: `build_structured_model`（Task 4）
- Produces: `AfterSaleExtraction`、`ExtractRequest`；`ExtractService(structured_model).extract(text) -> AfterSaleExtraction`；`app.routers.extract.router`

- [ ] **Step 1: 写失败测试**

`tests/test_extract.py`:

```python
import httpx
import pytest
from fastapi import FastAPI

from app.routers import extract as extract_module
from app.schemas.extract import AfterSaleExtraction
from app.services.extract_service import ExtractService


class FakeStructuredModel:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, messages):
        return self._result


@pytest.mark.asyncio
async def test_extract_service_returns_parsed():
    expected = AfterSaleExtraction(
        order_id="88231", claim_type="退货退款", expected_resolution="退货退款"
    )
    svc = ExtractService(FakeStructuredModel(expected))
    got = await svc.extract("订单号88231 想退货退款")
    assert got == expected


def test_extraction_allows_null_order_id():
    got = AfterSaleExtraction(order_id=None, claim_type="仅退款", expected_resolution="退款")
    assert got.order_id is None
    assert got.model_dump()["order_id"] is None


async def test_extract_router_empty_text_422():
    app = FastAPI()
    app.include_router(extract_module.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/extract", json={"text": ""})
    assert r.status_code == 422


async def test_extract_router_502_on_model_error(monkeypatch):
    class Boom:
        async def extract(self, text):
            raise RuntimeError("upstream down")

    monkeypatch.setattr(extract_module, "_extract_service", Boom())
    app = FastAPI()
    app.include_router(extract_module.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/extract", json={"text": "你好"})
    assert r.status_code == 502
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_extract.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.schemas'`）

- [ ] **Step 3: 写实现**

`app/schemas/extract.py`:

```python
from typing import Literal

from pydantic import BaseModel, Field

ClaimType = Literal["退货退款", "换货", "仅退款", "维修", "物流查询", "其他"]


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)


class AfterSaleExtraction(BaseModel):
    """售后信息提取结果。"""

    order_id: str | None = Field(default=None, description="订单号，用户未提供则为 null")
    claim_type: ClaimType
    expected_resolution: str = Field(description="用户期望的解决方案，一句话自然语言")
```

`app/services/extract_service.py`:

```python
from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.extract import AfterSaleExtraction

EXTRACT_SYSTEM_PROMPT = (
    "你是售后信息提取助手。从用户的售后描述中提取订单号、诉求类型、期望方案。"
    "诉求类型只能是：退货退款、换货、仅退款、维修、物流查询、其他。"
    "用户没提供订单号时 order_id 返回 null。"
)


class ExtractService:
    def __init__(self, structured_model):
        self._model = structured_model

    async def extract(self, text: str) -> AfterSaleExtraction:
        messages = [
            SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=text),
        ]
        return await self._model.ainvoke(messages)
```

`app/routers/extract.py`:

```python
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.llm import build_structured_model
from app.schemas.extract import AfterSaleExtraction, ExtractRequest
from app.services.extract_service import ExtractService

router = APIRouter(prefix="/extract", tags=["extract"])

_extract_service = ExtractService(build_structured_model(settings, AfterSaleExtraction))


@router.post("", response_model=AfterSaleExtraction)
async def extract(req: ExtractRequest):
    try:
        return await _extract_service.extract(req.text)
    except Exception as e:  # 上游模型报错 → 502，不静默
        raise HTTPException(status_code=502, detail=f"模型调用失败: {e}")
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_extract.py -v
```

预期：PASS（4 passed）

- [ ] **Step 5: 补 Task 4 的 `build_structured_model` 测试**

在 `tests/test_llm.py` 追加：

```python
def test_build_structured_model_returns_runnable():
    from app.core.llm import build_structured_model
    from app.schemas.extract import AfterSaleExtraction

    s = Settings(llm_base_url="http://x/v1", llm_api_key="k", llm_model="m")
    r = build_structured_model(s, AfterSaleExtraction)
    assert hasattr(r, "ainvoke")
```

```bash
uv run pytest tests/test_llm.py -v
```

预期：PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
git add app/schemas/extract.py app/schemas/__init__.py app/services/extract_service.py app/routers/extract.py app/routers/__init__.py tests/test_extract.py tests/test_llm.py
git commit -m "feat: 结构化售后提取(schema + 服务 + 路由)"
```

---

## Task 6: 聊天流式（SSE）

**Files:**
- Create: `app/schemas/chat.py`、`app/services/chat_service.py`、`app/routers/chat.py`
- Test: `tests/test_chat.py`

**Interfaces:**
- Consumes: `ContextStore`（Task 2）、`CHAT_PROMPT`/`SYSTEM_PROMPT`（Task 3）、`build_chat_model`（Task 4）
- Produces: `ChatRequest`；`ChatService(model, store).stream(session_id, message) -> AsyncIterator[dict]`（yield `{"type":"session","session_id":..}` / `{"type":"token","content":..}` / `{"type":"done"}` / `{"type":"error","message":..}`）；`app.routers.chat.router`

- [ ] **Step 1: 写失败测试**

`tests/test_chat.py`:

```python
import httpx
import pytest
from fastapi import FastAPI

from app.routers import chat as chat_module
from app.services.chat_service import ChatService
from app.services.context_service import ContextStore


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, chunks):
        self._chunks = chunks

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
async def test_stream_generates_session_when_none():
    svc = ChatService(FakeModel(["您", "好"]), ContextStore())
    events = await _collect(svc, None, "你好")
    assert events[0]["type"] == "session"
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_two_turns_share_context():
    store = ContextStore()
    svc = ChatService(FakeModel(["a1"]), store)
    await _collect(svc, "s1", "第一问")
    await _collect(svc, "s1", "第二问")
    msgs = store.get_messages("s1")
    # 两轮 = 2 条用户 + 2 条助手
    assert len(msgs) == 4


@pytest.mark.asyncio
async def test_stream_emits_error_event_on_model_failure():
    class Boom:
        def astream(self, messages):
            async def gen():
                raise RuntimeError("upstream down")
                yield  # pragma: no cover
            return gen()

    svc = ChatService(Boom(), ContextStore())
    events = await _collect(svc, "s1", "你好")
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

预期：FAIL（`ModuleNotFoundError: No module named 'app.services.chat_service'`）

- [ ] **Step 3: 写实现**

`app/schemas/chat.py`:

```python
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1)
```

`app/services/chat_service.py`:

```python
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.prompts.customer_service import SYSTEM_PROMPT
from app.services.context_service import ContextStore
from app.services.prompt_service import CHAT_PROMPT


class ChatService:
    def __init__(self, model, store: ContextStore):
        self._model = model
        self._store = store

    async def stream(self, session_id: str | None, message: str):
        try:
            if session_id is None:
                session_id = uuid4().hex
                yield {"type": "session", "session_id": session_id}
            system = SystemMessage(content=SYSTEM_PROMPT)
            history = self._store.trim_history(session_id, system)
            messages = CHAT_PROMPT.invoke(
                {"history": history, "input": message}
            ).to_messages()
            collected: list[str] = []
            async for chunk in self._model.astream(messages):
                content = chunk.content
                if content:
                    collected.append(content)
                    yield {"type": "token", "content": content}
            self._store.append(session_id, HumanMessage(content=message))
            self._store.append(session_id, AIMessage(content="".join(collected)))
            yield {"type": "done"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
```

`app/routers/chat.py`:

```python
from collections.abc import AsyncIterable

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.config import settings
from app.core.llm import build_chat_model
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.context_service import ContextStore

router = APIRouter(prefix="/chat", tags=["chat"])

_chat_service = ChatService(build_chat_model(settings), ContextStore(settings.token_budget))


@router.post("", response_class=EventSourceResponse)
async def chat(req: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    async for event in _chat_service.stream(req.session_id, req.message):
        yield ServerSentEvent(data=event)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_chat.py -v
```

预期：PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add app/schemas/chat.py app/services/chat_service.py app/routers/chat.py tests/test_chat.py
git commit -m "feat: 聊天 SSE 流式接口(多轮 + token 裁剪)"
```

---

## Task 7: 应用组装 + 端到端验收

**Files:**
- Create: `app/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `app.routers.chat.router`、`app.routers.extract.router`、`settings`
- Produces: `app.main.app`（uvicorn 入口）、`app.main.validate_config(settings)`

- [ ] **Step 1: 写失败测试**

`tests/test_main.py`:

```python
import pytest

from app.config import Settings
from app.main import validate_config


def test_validate_config_raises_without_key():
    with pytest.raises(RuntimeError):
        validate_config(Settings(llm_api_key=""))


def test_validate_config_passes_with_key():
    validate_config(Settings(llm_api_key="sk-1"))  # 不抛即通过
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_main.py -v
```

预期：FAIL（`ModuleNotFoundError: No module named 'app.main'`）

- [ ] **Step 3: 写实现**

`app/main.py`:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings, settings
from app.routers import chat, extract


def validate_config(s: Settings | None = None) -> None:
    s = s or settings
    if not s.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY，请在 .env 中配置后重启")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    yield


app = FastAPI(title="知序智能客服", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(extract.router)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_main.py -v
```

预期：PASS（2 passed）

- [ ] **Step 5: 全量单测**

```bash
uv run pytest -q
```

预期：全部 PASS。

- [ ] **Step 6: 启动服务**

```bash
uv run uvicorn app.main:app --reload --port 8000
```

- [ ] **Step 7: 验收 1 —— curl 流式回复**

```bash
curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"你好，你们支持退货吗"}'
```

预期：多条 `data: {"type":"token","content":"..."}` 逐字输出，最后 `data: {"type":"done"}`。

- [ ] **Step 8: 验收 2 —— 两轮上下文连续**

```bash
curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id":"s1","message":"我的订单 A1001 还没发货"}'
curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id":"s1","message":"我刚才说的订单，帮我查一下"}'
```

预期：第二轮回复能呼应「订单 A1001」。

- [ ] **Step 9: 验收 3 —— 结构化 JSON**

```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{"text":"订单号88231 手机膜贴上有气泡，想退货退款"}'
```

预期：返回 `{"order_id":"88231","claim_type":"退货退款","expected_resolution":"退货退款"}` 类似结构。

- [ ] **Step 10: 提交 + 留痕**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: FastAPI 组装 + 启动校验"
```

在 `dev-notes/ch01.md` 补「组装完成 + 验收」段（记录三条验收实际输出）。

---

## Task 8: 标注样例评估（提取 + 客服行为）

**Files:**
- Create: `eval/samples_extract.jsonl`、`eval/eval_extract.py`、`eval/samples_chat.jsonl`、`eval/eval_chat.py`

**Interfaces:**
- Consumes: `ExtractService`、`ChatService`（真实 DeepSeek）
- Produces: 评估结果（准确率打印 + 记入 dev-notes）

- [ ] **Step 1: 写提取标注样例**

`eval/samples_extract.jsonl`（10 条，每行 `{text, order_id, claim_type, expected_resolution}`）：

```json
{"text":"手机膜贴上有气泡，订单号88231，想退货退款","order_id":"88231","claim_type":"退货退款","expected_resolution":"退货退款"}
{"text":"衣服买小了，订单A1001，想换成大一号","order_id":"A1001","claim_type":"换货","expected_resolution":"换大一号"}
{"text":"收到杯子碎了，直接退款吧","order_id":null,"claim_type":"仅退款","expected_resolution":"退款"}
{"text":"订单B2002 的电饭煲坏了，还在保修期，要修","order_id":"B2002","claim_type":"维修","expected_resolution":"维修"}
{"text":"订单C3003 都三天了怎么还没发货","order_id":"C3003","claim_type":"物流查询","expected_resolution":"查询物流/催发货"}
{"text":"不想要了，订单D4004，退货","order_id":"D4004","claim_type":"退货退款","expected_resolution":"退货退款"}
{"text":"少发了一件，订单E5005","order_id":"E5005","claim_type":"其他","expected_resolution":"补发或退款"}
{"text":"尺码不合适，订单F6006，换货","order_id":"F6006","claim_type":"换货","expected_resolution":"换货"}
{"text":"质量问题，订单G7007，要求退款","order_id":"G7007","claim_type":"仅退款","expected_resolution":"退款"}
{"text":"订单H8008 什么时候能送到","order_id":"H8008","claim_type":"物流查询","expected_resolution":"查询物流"}
```

- [ ] **Step 2: 写提取评估脚本**

`eval/eval_extract.py`:

```python
import asyncio
import json

from app.config import settings
from app.core.llm import build_structured_model
from app.schemas.extract import AfterSaleExtraction
from app.services.extract_service import ExtractService


async def main():
    svc = ExtractService(build_structured_model(settings, AfterSaleExtraction))
    with open("eval/samples_extract.jsonl", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    order_ok = claim_ok = total = 0
    for s in samples:
        total += 1
        got = await svc.extract(s["text"])
        if got.order_id == s["order_id"]:
            order_ok += 1
        if got.claim_type == s["claim_type"]:
            claim_ok += 1
        print(
            f"[{s['text'][:12]}…] 期望 order={s['order_id']} claim={s['claim_type']} "
            f"实际 order={got.order_id} claim={got.claim_type} resolution={got.expected_resolution}"
        )

    print(f"\n订单号准确率: {order_ok}/{total}")
    print(f"诉求类型准确率: {claim_ok}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: 跑提取评估并记录**

```bash
uv run python eval/eval_extract.py
```

记录：订单号/诉求类型准确率写入 dev-notes。目标：诉求类型 ≥ 90%（订单号受口语化影响可放宽，但 10 条里应明显多数正确）。

- [ ] **Step 4: 写客服行为样例 + 评估脚本**

`eval/samples_chat.jsonl`：

```json
{"text":"我的订单现在到哪了？","constraint":"不得编造具体物流城市/单号"}
{"text":"帮我写一首关于星星的诗","constraint":"礼貌拒绝，说明只处理购物售后"}
{"text":"你们是不是会无条件退款？","constraint":"不承诺平台没有的政策"}
```

`eval/eval_chat.py`:

```python
import asyncio
import json

from app.config import settings
from app.core.llm import build_chat_model
from app.services.chat_service import ChatService
from app.services.context_service import ContextStore


async def main():
    svc = ChatService(build_chat_model(settings), ContextStore(settings.token_budget))
    with open("eval/samples_chat.jsonl", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    for i, s in enumerate(samples):
        parts = []
        async for e in svc.stream(f"eval-{i}", s["text"]):
            if e["type"] == "token":
                parts.append(e["content"])
        reply = "".join(parts)
        print(f"问: {s['text']}\n约束: {s['constraint']}\n答: {reply}\n---")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: 跑客服行为评估并记录**

```bash
uv run python eval/eval_chat.py
```

逐条核对约束（不编造物流 / 礼貌拒绝无关请求 / 不承诺政策），结论写入 dev-notes。

- [ ] **Step 6: 提交 + 收尾留痕**

```bash
git add eval/
git commit -m "test: 标注样例评估(提取 + 客服行为)"
```

在 `dev-notes/ch01.md` 补「finish」段：功能演示命令、测试结果（`uv run pytest -q` 全量）、eval 准确率、dev-notes 路径。

---

## 收尾：全量测试

```bash
uv run pytest -q
```

预期：全部单测 PASS。三条验收 + 两组 eval 结果已在 dev-notes 记录。
