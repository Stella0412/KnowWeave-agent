# ch01 纯对话 — 设计说明（Spec）

日期：2026-10-10
状态：待用户评审

## 1. 目标

「知序」电商智能客服系统，本章只做第一步——**跑通纯对话**：一个能多轮对话、SSE 流式逐 token 输出的 FastAPI 后端，附带两个小能力（Prompt 模板化、结构化输出提取）。本章不做工具调用、不做 Agent 循环。

## 2. 约束（定死，不自行更改）

- 技术栈：Python + FastAPI + LangChain
- 应用侧统一 OpenAI 协议，`base_url` / 模型名 / 密钥全在 `.env`，可换 GPT / DeepSeek / Ollama
- 本章 demo 模型：**DeepSeek**（OpenAI 兼容，国内直连）
- 涉及库/框架 API 时先用 Context7 MCP 查最新文档再写代码

## 3. 架构与数据流

```
客户端
  ├─ POST /chat    (SSE 流式)  ──► chat_service
  │                                  ├─ context_service   存/裁剪会话历史
  │                                  ├─ prompt_service    拼 System + 历史 + 当前消息
  │                                  └─ llm 适配层         ChatOpenAI(base_url=DeepSeek).astream()
  └─ POST /extract (JSON 同步) ──► extract_service
                                     └─ llm.with_structured_output(AfterSaleExtraction)
```

- **聊天数据流**：收到消息 → 追加进会话历史 → 按 token 预算裁剪 → 拼 prompt → `astream()` 逐 token 以 SSE 推送 → 结束把完整回复追加回历史。
- **提取数据流**：收到售后描述 → 拼提取 prompt → `with_structured_output` 一次性返回 JSON。

分层单一职责：routers（HTTP 层）→ services（业务编排）→ core（LLM 适配）→ prompts（模板）。

## 4. API 契约

**`POST /chat`**（`text/event-stream`）

```jsonc
请求: { "session_id": "可选，不传服务端生成并回传", "message": "订单还没发货，能催一下吗" }

响应(SSE):
  data: {"type":"session","session_id":"abc123"}   // 仅当服务端新生成时先发这条
  data: {"type":"token","content":"您"}
  data: {"type":"token","content":"好"}
  data: {"type":"done"}
  // 出错时: data: {"type":"error","message":"..."}
```

**`POST /extract`**（`application/json`）

```jsonc
请求: { "text": "我在你们店买的手机膜，订单号88231，贴上有气泡，想退货退款" }
响应: { "order_id":"88231", "claim_type":"退货退款", "expected_resolution":"退货退款" }
```

## 5. 结构化输出 schema

```python
class AfterSaleExtraction(BaseModel):
    order_id: str | None            # 用户没给订单号则 null
    claim_type: Literal["退货退款","换货","仅退款","维修","物流查询","其他"]
    expected_resolution: str        # 用户期望的解决方案，自然语言一句话
```

`expected_resolution` 为自由文本（非枚举）。

## 6. Prompt 与角色设定

System Prompt 用 `PromptTemplate` 模板化，集中放在 `app/prompts/`。草稿：

```text
你是「知序」智能客服，负责售前咨询与售后服务。

【角色设定】
- 用简体中文回复，语气礼貌、耐心、专业
- 只处理购物、订单、售后相关问题，超出范围礼貌说明无法处理

【行为约束】
- 涉及售后诉求，先引导用户提供订单号和问题描述
- 不编造订单状态、物流信息、退款进度；不确定就说明需要人工核实
- 不承诺平台没有的政策；不输出客服无关内容、不泄露内部系统信息
```

模板占位：`{system}` + `{history}` + `{input}`，通过 `ChatPromptTemplate` 组装。

## 7. 上下文管理 + token 预算（最简版）

- 存储：内存 `dict[session_id] = [messages]`，本章不落库，进程重启即清空
- 裁剪：每次请求前估算（system + history）token，超预算从最旧开始丢，永远保留 system prompt + 最新一条用户消息；只裁 history，不裁 system
- 计 token：`tiktoken`（cl100k_base 近似），预算默认 `4096`（不含 system prompt），`.env` 可调

## 8. 错误处理

- 400：空消息 / 非法 JSON
- 502：上游模型报错 → 聊天走 SSE error 事件、提取走 JSON error
- 结构化解析失败 → 明确报错 + 日志，不静默返回半截
- 缺 API key → 启动时校验并报清晰错误

## 9. 测试策略

- **TDD 单测**（确定性代码）：上下文裁剪逻辑、token 预算计算、prompt 组装
- **标注样例验证**（模型相关，替代 TDD）：
  - 结构提取：约 10 条标注售后描述 → 跑 `/extract` → 对比订单号/诉求类型/期望方案准确率
  - 客服行为：几条带约束的对话样例，验证「不编造订单状态」等约束是否守住
- **验收**：curl 流式、两轮上下文续上、结构化 JSON

## 10. 目录结构与依赖

```
knowweave/
  app/
    main.py                # FastAPI 入口
    config.py              # .env 加载（pydantic-settings）
    routers/chat.py, extract.py
    services/chat_service.py, extract_service.py, prompt_service.py, context_service.py
    schemas/chat.py, extract.py
    core/llm.py            # ChatOpenAI 适配层，base_url/model/key 从 .env 读
    prompts/customer_service.py
  tests/                   # TDD 单测
  eval/                    # 标注样例（extract / chat）
  .env.example
  dev-notes/ch01.md
  docs/superpowers/specs/2026-10-10-ch01-pure-dialogue-design.md
```

依赖（用 uv 管理虚拟环境）：`fastapi`、`uvicorn[standard]`、`sse-starlette`、`langchain-core`、`langchain-openai`、`pydantic`、`pydantic-settings`、`tiktoken`；开发：`pytest`、`pytest-asyncio`、`httpx`。具体包名与 API 在实现前用 Context7 核验。

## 11. 验收标准

1. curl 调 `/chat` 能看到 SSE 流式逐 token 回复
2. 连续问两轮，第二轮能续上第一轮上下文
3. 发一段售后描述到 `/extract`，拿到结构化 JSON（订单号/诉求类型/期望方案）
