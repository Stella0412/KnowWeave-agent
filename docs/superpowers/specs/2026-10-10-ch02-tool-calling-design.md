# ch02 Function Calling 工具链 — 设计说明（Spec）

日期：2026-10-10
状态：待用户评审

## 1. 目标

在 ch01 纯对话底座上，给「知序」客服加**查数据的能力**：用 Function Calling 让模型自己决定调什么工具，工具链长在客服聊天本身——用户在聊天页问一句就能触发。核心链路是「模型定工具 → 执行 → 回灌收敛」，**只做单轮**（模型调一次工具就收敛，不做多轮自动 Agent Loop）。

## 2. 约束（定死，不自行更改）

- 技术栈：FastAPI + SQLAlchemy + MySQL（Docker 起）+ LangChain（`langchain_core.tools` 的 `@tool` 装饰器）
- 建表 DDL 以仓库根 `建表语句.md` 为准，**不自行增删列**
- 应用侧统一 OpenAI 协议，demo 模型 **DeepSeek**（`deepseek-chat`）
- 商品/订单/物流三个工具内部随机造数，不接真实接口、不建表
- 只做单轮工具调用；不做 Agent Loop、不做向量检索/RAG
- 涉及库/框架 API 先 Context7 查最新文档再写代码

## 3. 架构与数据流

```
客户端
  └─ POST /chat (SSE 流式) ──► chat_service
        ├─ repository        读/写 conversations、messages、tickets、faq（DB 唯一真相源）
        ├─ context_service   内存 token 预算裁剪（退化为「裁剪计算器」，不再持有多轮状态）
        ├─ tools/registry    build_tools(db_session, conversation_id) → 五工具供 bind_tools
        ├─ tools/executor    参数校验 + 超时重试 + 错误兜底
        └─ llm 适配层         ChatOpenAI.bind_tools(...).ainvoke() 决策 / astream() 收敛
```

- **聊天数据流（单轮）**：收到消息 → 落 `messages(user)` → 读 `messages` 历史 + 内存裁剪 → `bind_tools` 决策（`ainvoke`）→ 命中工具则每个 tool_call 先推 `tool_status` 帧 → 落 `messages(assistant 带 tool_calls)` → executor 执行 → 落 `messages(tool)` → 回灌 → 最终回答 `astream` 逐 token 流出 → 落 `messages(assistant 全文)`。
- 分层单一职责：routers（HTTP）→ services（业务编排）→ tools（工具注册/执行）→ repositories + db（数据访问/ORM）→ core（LLM 适配）→ prompts（模板）。

## 4. 数据模型（以 `建表语句.md` DDL 为准）

四张表，全库 InnoDB / utf8mb4。建表顺序：conversations → messages / tickets（faq 无依赖）。

### conversations（会话壳）
- `id` BIGINT UNSIGNED 自增主键
- `user_id` VARCHAR(64) NOT NULL —— ch02 无鉴权，统一填固定值 `"anonymous"`（占位）
- `status` ENUM(`进行中`,`已转人工`,`已结束`) NOT NULL DEFAULT `进行中`（ch02 只触发 `进行中`→`已转人工`，`已结束` 无入口、留待后续）
- `created_at` / `updated_at`

### messages（消息流水）
- `id` BIGINT 自增主键
- `conversation_id` FK → conversations.id
- `role` ENUM(`user`,`assistant`,`tool`)
- `content` TEXT 可空（assistant 纯工具调用时为空）
- `tool_calls` JSON 可空（assistant 消息带的工具调用申请单 `[{name,args,id}]`）
- `tool_call_id` VARCHAR(64) 可空（tool 消息对应申请单 id）
- `created_at`

**落库语义**：assistant 想调工具 = role=assistant、content 空、tool_calls 存申请单；工具结果 = role=tool、content=结果、tool_call_id 对号；最终回答 = role=assistant、content=全文。

### faq（常见问答）
- `id` 自增主键 / `question` VARCHAR(512) / `answer` TEXT / `category` VARCHAR(64) / `created_at` / `updated_at`

### tickets（人工工单）
- `ticket_no` VARCHAR(32) 主键，格式 `T{YYYYMMDD}{3位当日序号}`（如 `T20260701008`），序号 = 当日已有工单数 + 1（零补 3 位）
- `conversation_id` FK → conversations.id
- `description` TEXT
- `ticket_type` ENUM(`售后`,`投诉`,`咨询`)
- `status` ENUM(`待处理`,`已处理`) NOT NULL DEFAULT `待处理`
- `created_at`

### session_id ↔ conversation_id 映射

conversations.id 是自增数字，而 ch01 前端客户端自造 uuid。为不背离 DDL（不加列），改为：**前端首条消息 `session_id` 传 null → 后端 INSERT 会话拿数字 id → 经 `{"type":"session","session_id":"<数字id>"}` 回传 → 前端存下、后续带上**。API 层 `session_id` 语义 = `str(conversations.id)`。

## 5. 工具定义（LangChain `@tool`）

- `query_order(order_id: str) -> str`：mock 订单数据（状态/金额/商品），回显 `order_id`
- `query_product(product_name: str) -> str`：mock 商品数据（价格/库存/规格），回显 `product_name`
- `query_logistics(order_id: str) -> str`：mock 物流数据（承运商/当前节点/预计送达），回显 `order_id`
- `query_faq(keyword: str) -> str`：`WHERE question LIKE %kw% OR answer LIKE %kw%` 查 faq 表，命中返回问题+答案，未命中返回「未找到相关常见问题」
- `create_ticket(description: str, ticket_type: Literal["售后","投诉","咨询"]) -> str`：写 tickets 表返回 ticket_no，并把 conversation.status 置 `已转人工`

**依赖注入**：`query_faq` 注入 DB 会话、`create_ticket` 注入 DB 会话 + 当前 `conversation_id`；模型只填业务参数（`keyword`/`description`/`ticket_type`），会话 id 由编排层注入，不让模型瞎填。工具列表每个请求按当前会话构建。

## 6. 工具基础设施

- **registry**：`build_tools(db_session, conversation_id) -> list[BaseTool]`，返回五工具供 `model.bind_tools`。
- **executor**：`execute(tool_call, tools) -> ToolMessage`，三步：
  1. 用 `tool.args_schema`（Pydantic）校验入参，非法则返回错误 ToolMessage；
  2. `asyncio.wait_for(tool.ainvoke(args), timeout=settings.tool_timeout)` + 重试 `settings.tool_retries` 次；
  3. 工具抛错/超时 → 转成 ToolMessage 错误文本，**不崩流**，让模型兜底答复「查询失败」。

## 7. 聊天链路 + SSE 协议（单轮）

```
落 messages(user) → 读 messages 历史(按 created_at) + 内存 token 裁剪
  → model.bind_tools(五工具).ainvoke() 决策
  → 命中工具: 每个 tool_call 推 tool_status 帧 → 落 messages(assistant 带 tool_calls)
    → executor 执行 → 落 messages(tool) → 回灌
  → 最终回答 astream 逐 token 流出 → 落 messages(assistant 全文)
```

SSE 事件在 ch01 基础上**新增一种**：

```jsonc
data: {"type":"session","session_id":"1"}                       // 仅新会话回传数字 id
data: {"type":"tool_status","name":"query_logistics","status":"running"}  // 工具执行前推
data: {"type":"token","content":"您"}
data: {"type":"done"}
// 出错: data: {"type":"error","message":"..."}
```

前端收到 `tool_status` 就在气泡里挂「🔧 调用了 XX」小徽章（Vibe Coding 做样式）。

## 8. 错误处理

- 工具超时/报错 → 转 ToolMessage 错误文本，模型兜底答复，流不中断。
- DB 未起/连不上 → SSE `error` 事件，清晰报错。
- `query_faq` 查不到 → 工具返回「未找到」，模型如实说明（**不编造政策**，System Prompt 加约束）。
- 缺 LLM key → 沿用启动校验（ch01 `validate_config`）。
- 建表/灌数据由 `app.db.seed` 幂等执行；lifespan 内 `create_all`（幂等建表），DB ping 失败仅打清晰日志、不强制崩溃。

## 9. 测试策略

- **TDD 单测**（确定性，不碰真实模型/MySQL）：
  - repository 四表读写与状态更新（create_ticket → conversation 状态置 `已转人工`）
  - `query_faq` 的 LIKE 匹配（命中/未命中）
  - `create_ticket` 写库 + ticket_no 生成格式
  - 三个 mock 工具返回形状 + 回显入参
  - executor 的参数校验/超时重试/错误兜底（用 fake tool）
  - chat_service 编排（fake model 返回带 tool_calls 的 AIMessage，验证「决策→执行→回灌→落库→流式」与单轮收敛）
  - DB 相关测试用 **aiosqlite 内存库**（生产仍 MySQL）
- **标注样例评估**（模型相关，替代 TDD）：`eval/samples_tools.jsonl`（用户问句 → 期望工具，覆盖订单/商品/物流/FAQ/工单 + 邮费漏召回），跑真实 DeepSeek 算工具选择准确率；最终回答不编造抽查。
- **验收**：浏览器 + curl 三条验收标准。

## 10. 目录结构与依赖

```
knowweave/
  app/
    config.py            # 增补：MySQL 连接参数 + tool_timeout / tool_retries
    db/
      __init__.py
      base.py            # DeclarativeBase
      engine.py          # create_async_engine + async_sessionmaker + get_session
      models.py          # FAQ / Conversation / Message / Ticket
      repository.py      # ChatRepository：会话/消息/工单/FAQ 读写
      seed.py            # 建表 + 灌 faq 测试数据（幂等；覆盖退货/发货/换货/售后等，**不含邮费/运费**，为验收③保留漏召回）
    tools/
      __init__.py
      business.py        # 5 个 @tool（含依赖注入）
      registry.py        # build_tools
      executor.py        # 校验 + 超时重试 + 错误兜底
    services/chat_service.py  # 改造：单轮工具编排 + 落库
    prompts/customer_service.py  # 增补「何时用工具、查不到别编造」约束
  static/index.html      # 改造：工具轨迹徽章（Vibe Coding）
  docker-compose.yml     # MySQL 8
  eval/samples_tools.jsonl + eval/eval_tools.py
  tests/                 # TDD 单测
  dev-notes/ch02.md
  docs/superpowers/specs/2026-10-10-ch02-tool-calling-design.md
```

依赖新增：`sqlalchemy[asyncio]`（含 greenlet）、`aiomysql`；开发测试加 `aiosqlite`。`@tool` 已在 `langchain_core.tools`，无需引整包 `langchain`。

MySQL 连接从 `.env` 读（`mysql_host/port/user/password/db`），拼 `mysql+aiomysql://...`。

## 11. 验收标准

1. 浏览器打开聊天页，问「订单 1001 的物流到哪了」→ 气泡带工具徽章(query_logistics)，并按返回结果作答。
2. 问「退货政策是什么」→ `query_faq` 命中并作答。
3. 换说法问「邮费是多少」→ `query_faq` 查不到，如实说未查到（**漏召回为预期**，记入 dev-notes 留给 RAG 升级）。
