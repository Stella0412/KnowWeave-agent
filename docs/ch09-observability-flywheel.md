# ch09 · 观测与飞轮闭环：跟踪与验收

本章把 👍/👎 反馈真正落库，👎 连同 citations 快照入池；池中问题按「归一化指纹精确查重 +
向量语义归并」聚合；审核页支持补答案（入库 + Milvus 向量化）、驳回、重开；每轮对话的
route/intent/状态/耗时/首 token/token/成本落到 `turn_metrics`，全图 LLM/工具链路进 Langfuse，
`/observability` 页面用 MySQL 聚合、不依赖 Langfuse。以下命令默认在仓库根目录（D:/knowweave）
执行，Python 3.14，全部经 `uv run`。

## 1. 架构与四个验收页面

```text
聊天页 /chat (SSE)
  └─ ChatService.stream/run
       ├─ thread_config(conversation_id, turn_token)
       ├─ graph（编译时挂 TurnMetricsHandler + Langfuse）
       │    ├─ LLM 调用 ──► 回调聚合 token/成本/首 token
       │    ├─ 工具调用 ──► 显式 Langfuse span + tool_audit_logs.turn_token
       │    └─ RAG pipeline ──► embedding/rerank 阶段观测（估算）
       ├─ _persist_new → message_id
       ├─ aupdate_state({citations, citations_by_message_id})
       └─ finish_turn → turn_metrics
              │
              ├─ POST /api/feedback（👍/👎）
              │     ├─ answer_feedback（message_id 唯一）
              │     └─ 👎 → FlywheelService.record_event（含 citations）
              │
              └─ /review 静态页 + REST
                    ├─ 队列/详情（来源、次数、快照）
                    ├─ 补答案 → staging(kept) → knowledge_chunks(pending)
                    │            → 进程内 Milvus 向量化 → 关闭问题
                    ├─ 驳回 → 已忽略；重开 → 待处理
                    └─ 补向量化
```

批处理（`make`）：`merge`（指纹回填 + 精确/语义归并，可 `--dry-run`）、`gap-review`（审核页命令行备份）、
`vectorize`（清 pending 向量，需停应用）、`eval-merge`（语义阈值评估）、`eval-live`（`CH09_LIVE=1` 真实闭环）。

四个验收页面：

| 页面 | URL |
|---|---|
| A 聊天页 | `http://localhost:8000/` |
| B 飞轮审核 | `http://localhost:8000/review` |
| C 观测与成本 | `http://localhost:8000/observability` |
| D Langfuse | `http://localhost:3000` |

## 2. 前置与 MySQL 迁移顺序

1. MySQL 迁移：**按顺序**手工执行 `sql/ch09-flywheel.sql`、`sql/ch09-observability.sql`（两份文件顶部已含
   `SET NAMES utf8mb4;`）。`ALTER` 是一次性的，重复执行报错属预期。应用启动时会跑
   `Base.metadata.create_all`（app/main.py:69）补齐缺失的**表**，但**不会**补列/加列/`ALTER`——
   缺列时仍会在池/指标读写处报错，需先手工执行上述 `sql/ch09-*.sql` 再起服务。
2. 回填与归并：先 `make merge ARGS="--dry-run"` 检查回填/归并计划（打印将合并的根对、跨状态跳过项、
   次数变化，**不写库**）；确认后再 `make merge` 真正回填指纹并对历史行归并。合并不可自动回滚，正式运行前
   建议备份 `low_confidence_questions` 表。
3. 起观测栈：`make langfuse-up`（自动补齐缺失的本地 Langfuse 密钥，首次 1–2 分钟）。
4. 起应用：`make dev`。
5. 确认 `.env` 中 LLM/embedding/rerank key 已存在（只验存在，不打印值）。
6. 过程记录写 `dev-notes/ch09.md`（gitignore，不入库）；**不把用户原话/密钥写进任何入库文档**。

## 3. 反馈与快照语义

- **只允许评价最新一条已完成助手消息**：服务端比较 `message_id` 与当前最新 assistant 消息 id，
  不是最新 → 409（不做「把最新一轮快照挂到旧消息」的错配）。历史消息补评不在本章目标内；
  前端只在最新一条完成消息上挂反馈按钮，出现新助手消息后把上一条按钮置为只读。
- 点踩时从该会话 checkpointer state 取 `citations_by_message_id[str(message_id)]` 快照一并落库；
  取不到就留空（`None`）。同一轮可能有拒答 + 点踩两个信号，各自带快照。
- **快照三态**（审核页 `evidence_state`）：
  - `missing`：没有快照（旧消息 / state 缺失 / 写失败）→ 显示「无快照」；
  - `empty`：快照为 `[]`（该轮没走检索或召回为空）→ 显示「该轮无检索片段」；
  - `has_snippets`：有片段 → 正常展示。
- 拒答轮若检索过（含低分召回/自评拒答）保留当时检索到的片段；只有空查询 / 召回为空 / 检索异常时
  快照才为空，不编造片段。快照上限 20 条、单条 `answer` 截 2000 字符、序列化后超 64KB 从尾部丢弃。
- **不自动复活**：命中 `已关闭`/`已忽略` 主问题时，只 `duplicate_count += 1`、`last_seen_at = now` 并插入
  从属行，状态不变；只有人工「重新打开」才能回到 `待处理`。

## 4. 归并语义（批处理）

`uv run python -m app.jobs.low_conf_merge [--dry-run] [--threshold 0.92] [--limit 2000]
[--batch-size 64] [--skip-semantic] [--require-semantic] [--json]`，分三阶段：

1. **回填**：`fingerprint=''` 的历史行计算指纹。
2. **精确归并（只在同状态根之间）**：按 `(fingerprint, status)` 分组，不同状态永不合并。组内选 canonical：
   有 `knowledge_chunk_id` 的优先 → `duplicate_count` 高者 → `id` 小者。loser 标记 `已归并`、`merged_into_id`
   指 canonical，`duplicate_count` 降为 1；canonical 状态不变（归并簿记不是处置结论，不产生复活）。
   canonical 无 chunk 且恰好一个 loser 有则继承其 chunk id 与 resolution（仅元数据）；多个不同 chunk id 时
   保留第一个并打 warning。
3. **语义归并（只在同状态根之间）**：embedding 余弦 + 并查集，`sim >= threshold` 成候选簇，映射到根后
   再按根状态分组，同状态才合并；跨状态只记 `skipped_cross_status`，不写库。

- `--dry-run`：打印计划，不写库。
- `--skip-semantic` 与 `--require-semantic` **互斥**，同时给 → 退出码 1。
- 退出码：0 成功；1 致命；2 `--require-semantic` 且语义阶段降级（embedding 失败）。
- 阈值 `low_conf_merge_threshold = 0.92` 保守初值；`make eval-merge` 输出 0.85–0.98 扫描并给出推荐阈值
  （写进文档，不自动改配置）。
- **实测推荐（2026-10-26，当前样本）**：`make eval-merge` 扫描在 **0.85** 处最优，precision 1.000 /
  recall 0.400 / f1 0.571（对比现状 0.92：precision 1.000、recall 0.100）。0.85 更激进且扫描区间内
  precision 均为 1.000，但样本量小、recall 上限受正例数限制，故**配置暂维持 0.92 不变**，待样本增大后复核；
  本记录不自动改配置。

## 5. Milvus Lite 锁与向量化

Milvus Lite 是**单进程独占锁**：审核页补答时应用与 CLI 不能同时向量化。因此：

- 应用运行中只能走审核页「补向量化」或 `/review/api/vectorize`（进程内全局 `asyncio.Lock`）；
  拿不到锁不排队，返回 202 `vectorize_status=pending`，可重试。
- `make vectorize`（`uv run python -m app.jobs.vectorize_kb`）清 pending 向量，**需先停应用**，
  否则因独占锁失败、chunk 留 pending。
- 补答闭环：`submit_answer` 事务内入库（staging kept → chunk pending → 根 `已关闭`），事务外向量化；
  向量化失败不影响关闭，chunk 留 pending 由「补向量化」/`make vectorize` 重试。

## 6. token 与成本口径

- LLM：`prompt/1e6 * input_price + completion/1e6 * output_price`；embedding/rerank 只有 input 估算 token
  乘 input 单价。
- `cost_cny` 保留 6 位小数。
- **`usage_estimated`**：任一调用缺 usage（模型不返回）或来自阶段估算（回调看不到 embedding/rerank）就为
  true；`TokenCounter` 兜底估算。
- **未配单价**：`config/model_prices.json` 中 `input_per_million`/`output_per_million` 为 `null` 的模型记 0 成本，
  进入 `unpriced_models`；页面显示「未配单价」，**不伪造金额**。模型表 `priced = 该模型全部调用都有单价`。
- 观测页指标口径（`days` 只允许 1/7/30）：`refusal_rate = 有非空 refusal_source 轮次 / turns`；
  `error_rate = (error + cancelled) / turns`（`interrupted` 单独展示，不计入错误率）；`satisfaction.rate =
  up/(up+down)`，分母为 0 显示「—」；P50/P95 只对非 NULL 值计算；窗口最多取最近 5000 行近似。

### 6.1 意图成本账（Langfuse）

- **标签**：`IntentTaggingCallbackHandler` 在 `classify_intent` 结束后把 `intent:<x>` 合并进此后创建的观测的
  `langfuse.trace.tags`（链、生成、工具与当轮根 span `handle-chat-turn` 都带）；分类之前的指代消解与分类调用
  不带标签。口径是**只统计意图确定之后的开销**，分类前的花销单列「意图未定」，不摊进任何意图。
- **数据源**：Langfuse v2 Metrics API（`GET /api/public/v2/metrics`，`observations` 视图）按 `tags` 分组取
  `totalTokens`/`totalCost`；请求数取观测名 `handle-chat-turn` 的 `count`（一轮一个根 span）；按
  `environment` 过滤只算当前环境。v4 events_only 模式的 `traces` 视图不可用，不能按 trace 直接计数。
- **产物与重跑**：产物落 `data/cost_report/latest.json`（`data/*` 已忽略）。页面只读产物、不直连 Langfuse；
  重跑走页面按钮（`POST /observability/api/cost-report/rerun`，窗口 1/7/30，默认 7 天）或
  `make cost-report ARGS="--days 7"`。
- **成本来源**：金额是 Langfuse 在 ingestion 时按模型定义（`POST /api/public/models` 的 `pricingTiers`）
  算出的值；模型没定价、或 trace 在定价前入库，成本就是 0。本地 `config/model_prices.json` 的 `null`
  不影响账本，但页面上方「模型用量与成本」仍会显示 0 成本。

## 7. Langfuse 启动/关闭与版本配对

- 启动：`make langfuse-up`（先 `langfuse-env` 补齐缺失本地密钥，再 `docker compose -f docker-compose.langfuse.yml up -d`，
  等待 `/api/public/health`）。停止：`make langfuse-down`（保留数据卷，重置需人工 `docker compose down -v`，破坏性）。
  日志：`make langfuse-logs`；健康检查：`make langfuse-status`。
- **版本配对（2026-10-26 实测）**：web/worker `docker.langfuse.com/langfuse/langfuse:4`（实测 4.70.0）；
  ClickHouse `25.12`（v4 需要 `input_format_json_throw_on_bad_escape_sequence`，24.3 不支持）；PostgreSQL
  `15-alpine`；Redis `7-alpine`；MinIO `RELEASE.2024-05-28T17-19-04Z`（Docker Hub 已删除 `latest`，必须固定版本）。
  v4 以 events_only 模式运行，旧 `/api/public/traces` 不可用：trace 经 OTel 写入 MinIO 的 `langfuse` 桶，再进
  ClickHouse `events_full`；compose 启动 MinIO 前执行 `mkdir -p /data/langfuse` 建桶。ClickHouse 健康检查用
  `127.0.0.1`（容器内 `localhost` 可能解析到未监听的 ::1）。独立 compose 项目名 `knowweave-langfuse`，`down`
  不碰无关容器。建议 ≥4GB 可用内存，首次启动 1–2 分钟。
- **密钥配对**：compose 把 `.env` 的 `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` 同时注入为
  `LANGFUSE_INIT_PROJECT_PUBLIC_KEY`/`LANGFUSE_INIT_PROJECT_SECRET_KEY`，保证 SDK 与自托管项目用同一套 key
  （否则应用侧 401）。`init_langfuse_env` 补齐缺失的键，并把明显的占位值
  （空值、`<…>` 包裹、`your-…` 前缀、`changeme`/`change-me`）替换为生成值；**绝不覆盖真实已有值**、绝不打印值。
- **隐私仅本地**：Langfuse 为本地自托管，trace 含 prompt/回答正文；禁止把 `LANGFUSE_BASE_URL` 指向云端并灌入
  真实用户数据。`turn_metrics` 不含正文；审核页含用户原话/答案，仅本地开发使用，不暴露公网、无鉴权。

## 8. 内容边界

入库文档与代码中不含：用户原话、答案正文、聊天样本、任何 API 密钥（LLM/embedding/rerank/Langfuse）。
密钥只放 `.env`，检查只验非空、不打印值；`dev-notes/`、`docs/superpowers/` 不入库。

## 9. 验收运行命令

**离线（本机可执行，无需 MySQL/Langfuse/真实模型）：**

```bash
uv run pytest tests/test_ch09_live.py -q     # 默认 skip，预期 1 skipped
uv run pytest -q                             # 全量回归，预期 872 passed / 7 skipped
uv run python -m compileall app tests eval   # 语法编译全通过
make test                                    # 与 uv run pytest -q 同一套件
```

**真实闭环（需先停应用，Milvus Lite 单进程锁；需 MySQL + Langfuse + 真实模型/embedding）：**

```bash
CH09_LIVE=1 uv run pytest -q tests/test_ch09_live.py
# 或 make eval-live
```

闭环路径：跑一条注定答不出的唯一问题 → 拒答 → `record_event_isolated` 建根 → `submit_answer` 补答并
向量化 → 再问同一问题 → 回答并引用新 chunk → 删除测试 chunk。若 Milvus 客户端不支持删除，保留 chunk
并在验收报告中注明。

**页面验收（A/B/C/D 与批处理门禁，见下节；需 MySQL/Langfuse/应用依次就绪）：**

```bash
make merge ARGS="--dry-run"   # 先看计划，不写库
make merge                    # 确认后真正回填指纹并归并
make langfuse-up              # 起观测栈
make dev                      # 起应用
make cost-report              # 意图成本账产物（也可在观测页点「在页面上重跑」）
```

### 9.1 A/B/C/D 页面清单（spec §14.2）与批处理门禁（spec §14.3）

> 状态：**已于 2026-10-26 在本机完成验收（PASSED）**。MySQL（127.0.0.1:3308）、Langfuse v4.70.0
> （127.0.0.1:3000）与应用（127.0.0.1:8000）全部实跑；清单逐项勾选，证据与修复记录见 §9.2。
> 验收只清理了本次合成的数据（反馈、池行、chunk、向量），历史归并结果与指标保留。

**A 聊天页 `http://localhost:8000/`**

- [x] 提问正常政策 → 回答，👍 成功提示。
- [x] 提问知识库缺失问题 → 拒答；👎 → 成功，`/review` 中出现该问题。
- [x] 拒答轮若检索过（含低分召回/自评拒答），审核详情能看到当时检索到的片段；只有空查询/召回为空/检索异常时
      显示「无快照」，不编造片段。

**B 飞轮审核 `http://localhost:8000/review`**

- [x] 待处理队列显示次数、来源、快照、点踩评论。
- [x] 补答前另开会话重复同一问法（拒答/点踩）→ 同一主问题次数增加，状态仍是待处理。
- [x] 补答案 → 状态已关闭、chunk id、向量化 done。
- [x] 回到聊天问同一问题 → 回答并引用新 chunk。
- [x] 「重开」已关闭问题 → 回到待处理；再次关闭。关闭后不自动复活（细节由单测覆盖）。
- [x] 驳回一个未补答的问题 → 已忽略；再开会话问同一问法并点踩 → 计数增加、状态仍是已忽略（在「已忽略」标签页可见）；
      重开 → 待处理。
- [x] 「补向量化」能把 pending 清零。

**C 观测与成本 `http://localhost:8000/observability`**

- [x] 轮次数、满意度、拒答率、错误率、P50/P95、首 token、token 总量、成本/未配单价提示。
- [x] 轮次表能看到 route/intent/status/tokens/latency/trace_id。
- [x] 工具表能看到工具调用、失败、平均耗时、重试。
- [x] Langfuse 停掉时页面仍可用（数据来自 MySQL）。

**D Langfuse `http://localhost:3000`**

- [x] 用 `.env` 中的邮箱/密码登录（密码不打印）。
- [x] 能看到 `chat-turn` trace，含 LLM generations、工具 span、session=会话 id。
- [x] 轮次表的 trace_id 能在 Langfuse 定位到同一条 trace（若配置了 project id 则深链）。

**批处理质量门禁（spec §14.3，不替代页面验收）**

- [x] `make merge ARGS="--dry-run"` 输出计划；去掉 dry-run 后次数/主问题变化符合 §6。
- [x] `make eval-merge` 输出阈值扫描；推荐阈值写入文档，不自动改配置。
- [x] `make eval-live` 在停应用后通过闭环测试。
- [x] `make gap-review ARGS="--list"`、`ARGS="--metrics"` 能正常输出（CLI 辅助入口 smoke）；
      补答闭环已在页面 B 完成，不要求 CLI 再跑一遍。
- [x] `make vectorize` 在停应用后能把 pending 清零；应用运行时走 `/review/api/vectorize`（页面「补向量化」）。

**通过标准（spec §14.4）**：上述页面全部可用，闭环 A→B→A 成功；全量测试通过，无新增 skip（除门控 live）；
不打印任何密钥；`.env` 未被无意修改（除 `init_langfuse_env` 补齐的 Langfuse 本地键）；不改动既有 LLM/MySQL/embedding 值。

### 9.2 验收结果（2026-10-26 实测）

**环境**：MySQL 8（`knowweave-mysql`，宿主 3308）；Langfuse 自托管 v4.70.0（3000，events_only）；应用 uvicorn
（127.0.0.1:8000）；真实 LLM + embedding + rerank。`sql/ch09-flywheel.sql`、`sql/ch09-observability.sql`
在真实 MySQL 上按序执行成功（中文枚举按 utf8mb4 落库，`HEX` 校验无乱码）。

| 项目 | 命令 | 结果 |
|---|---|---|
| 全量回归 | `uv run pytest -q` | **807 passed / 7 skipped**（skip 均为既有门控/环境） |
| 语法编译 | `uv run python -m compileall -q app tests eval` | 通过 |
| 真实闭环 | `make eval-live` | **1 passed**（拒答→入池→补答+向量化→再问引用新 chunk→清理） |
| 归并计划 | `make merge ARGS="--dry-run"` | 回填 0 / 精确 0 / 语义 0（历史已归并干净），不写库 |
| 阈值评估 | `make eval-merge` | 见 §4：0.85 最优（P 1.000 / R 0.400 / F1 0.571），配置保持 0.92 |
| CLI 门禁 | `make gap-review ARGS="--list"` / `--metrics` | 正常输出（9 个根问题、23 条信号） |
| 向量补齐 | 停应用后 `make vectorize` | 补齐 **0 条**（无 pending） |
| Langfuse 降级 | 停 `langfuse-web`/`worker` 后访问 C 页 | `/observability` 与 `/api/summary`、`/api/turns` 仍 200，数据来自 MySQL |
| 忽略/重开 | 审核 API：驳回 → `status=已忽略` 且出现在「已忽略」标签页 → 重开 → `待处理` | 通过 |

A/B/C/D 页面 22 项清单已全部逐项实测勾选：👍/👎 落库与拒答快照三态、重复问法计数不复活、补答关闭并
向量化、再问引用新 chunk、重开/驳回/忽略标签页、观测页五项指标与轮次/工具表、Langfuse 登录后可见
`chat-turn` trace 及 session/工具 span、轮次表 `trace_id` 与 Langfuse 中同一条 trace 对应。

**验收中发现并修复的问题**（均已回归覆盖）：

1. `TurnMetricsHandler` 未继承 `BaseCallbackHandler`，回调管理器读取 `run_inline` 直接抛错，**每轮对话 500**；
   改为继承并在 `tests/test_ch09_handler.py` 增加兼容性回归。
2. 真实模型上报的序列化参数用 `model_name`，handler 只读 `model`/`name`，导致模型名记成 `ChatOpenAI`、
   价格匹配不上；改为依次读取 `model`/`model_name`/`name`，并补归属与计价回归。
3. Langfuse v4 events_only 需要 MinIO 预先存在 `langfuse` 桶，否则事件 `NoSuchBucket`、页面无 trace；
   compose 的 MinIO 启动命令改为先 `mkdir -p /data/langfuse`。
4. ClickHouse 健康检查用 `localhost` 在容器内解析到未监听的 `::1`，改为 `127.0.0.1`；v4 要求
   `input_format_json_throw_on_bad_escape_sequence`，ClickHouse 24.3 不支持，升到 25.12。
5. `sql/ch09-observability.sql` 的 `ADD COLUMN KEY idx_turn_token` 为非法语法，改为独立 `ADD KEY`。
6. Windows 下 Cygwin `make` 无法执行 Docker Desktop 的 sh 兼容层，Makefile 增加 `DOCKER ?= docker.exe`。

**已知无害现象**：CLI 退出时偶见 aiomysql `Connection.__del__` 的「Event loop is closed」告警，不影响
执行结果；验收期间的 3 条 `status=error` 轮次来自修复前的真实 500，保留为证据未删除。

**数据清理**：仅清理本次合成的验收数据（反馈、池行、chunk、向量）；历史归并结果、`turn_metrics`、
`tool_audit_logs` 保留。验收过程不打印任何密钥，`.env` 仅由 `init_langfuse_env` 补齐本地 Langfuse 键。
