# ch08 · 即插即用工具系统使用说明

本章把客服的工具能力收敛为统一注册/执行引擎：内置工具（订单/商品/FAQ/退款/建单）+ 两个独立
MCP Server（物流 `logistics`、售后 `after_sales`），经官方 `langchain-mcp-adapters` 动态发现，
再按本地权限文件逐工具授权。写工具（建单）必须走服务端可信确认；MCP 读工具支持「Server 新工具
热发现 + 权限热加载」。以下命令默认在仓库根目录执行。

## 1. 启动三个服务

```powershell
# 物流 MCP Server（端口 8101）
.venv/Scripts/python.exe -m mcp_servers.logistics

# 售后 MCP Server（端口 8102）
.venv/Scripts/python.exe -m mcp_servers.after_sales

# 客服 App（端口 8000）
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

App 依赖 MySQL 与 LLM 密钥。联调验收使用临时 Docker 容器 `knowweave-ch08-mysql`（只动临时库，不改
`.env`），用环境变量覆盖即可（`Settings` 按字段名大小写不敏感地解析环境变量）：

```powershell
$env:MYSQL_HOST="127.0.0.1"; $env:MYSQL_PORT="3307"; $env:MYSQL_USER="root"; `
$env:MYSQL_PASSWORD="knowweave123"; $env:MYSQL_DB="knowweave"
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 2. 演示命令

```powershell
# 双 Server 发现并调用（自己启动辅助进程，独立端口 + 临时权限文件，用完自动清理）
.venv/Scripts/python.exe -m scripts.demo_ch08 --mode mcp

# 新工具热发现：只重启 logistics Server + 更新临时权限文件后新公开名出现
.venv/Scripts/python.exe -m scripts.demo_ch08 --mode hot-reload

# 工单确认 / 取消（对 --base-url 指向的运行中 app 走真实 HTTP 链路）
.venv/Scripts/python.exe -m scripts.demo_ch08 --mode ticket-confirm --base-url http://127.0.0.1:8000
.venv/Scripts/python.exe -m scripts.demo_ch08 --mode ticket-cancel  --base-url http://127.0.0.1:8000

# 受控慢 MCP 读工具触发超时分类（绝不写真实工单）
.venv/Scripts/python.exe -m scripts.demo_ch08 --mode timeouts
```

演示进程只启动/停止它自己创建的辅助进程（MCP 临时 Server），打印机器可读证据
（PID、公开工具名、call_id、HTTP 状态、工单行/审计行、耗时），不新增任何公开注册端点。

## 3. 工单确认 / 取消 / 超时验收步骤

1. 按第 1 节起三个服务（App 指向临时 MySQL）。
2. 发一句明确建单请求，例如「帮我建个工单，反馈订单 1001 发货太慢」；`/chat` SSE 会先吐
   `tool_status`，再吐 `actions`（内含 `ticket_preview`，带 `confirmation_id`）后停在
   `done`，此时**不落工单、不落成功审计**。
3. **确认**：`POST /chat/ticket/resume`，body `{"session_id","confirmation_id","decision":"confirm"}`。
   预期：HTTP 200；`tickets` 恰好 +1（工单号 `T8` + 30 位 hex，稳定可重放）；`tool_audit_logs`
   恰好 +1 条 `create_ticket / builtin / 成功`。同一 `confirmation_id` 重复恢复返回 409，不追加写入。
4. **取消**：`decision:"cancel"`。预期：HTTP 200；`tickets` 零新增；审计 +1 条
   `权限拒绝`（`error_message="用户取消确认"`）。
5. **失效 / 越权**：过期 `confirmation_id` 或会话不匹配返回 409，无任何写入；多余字段（如试图
   注入 `description`）返回 422。
6. **超时**：用 `--mode timeouts` 演示；真实建单路径不注入故障。写工具默认不重试
   （`retry_count` 恒为 0），读工具超时/传输故障按 `tool_retries` 有限重试。

## 4. 新增工具 + 授权热加载

工具能力完全由 MCP Server 声明，App 不注册、不改代码：

1. 在某个 MCP Server（如临时 logistics 源码）新增一个读工具函数并重启**该 Server 进程**。
2. 编辑 `config/tool_permissions.json`（或临时权限文件），在对应 server 下加一行
   `"新工具原始名": "read"`。
3. App **进程不动**，下一轮对话自动 `refresh()` 重新发现并加载授权，新公开名
   `{server}_{原始名}` 立即对模型可见；未授权的新工具即使已被发现也**不暴露**。

反向亦然：从权限文件移除某行，下一轮该工具即不可见（权限热撤销）。

## 5. 权限配置样例的实际解析说明

`config/tool_permissions.json` 的解析由 `app/tools/policy.py::PolicyStore.load()` 完成，
语义如下（勿按「别名」理解，按 `server + 原始工具名` 精确匹配）：

```json
{
  "mcp": {
    "logistics":   {"query_logistics": "read"},
    "after_sales": {"query_warranty": "read", "query_return_progress": "read"}
  }
}
```

- 根必须是 `{"mcp": {server: {tool_original_name: "read"|"write"}}}`，任一结构非法 → 整体拒绝
  **所有** MCP 工具（不沿用旧授权，fail-closed）。
- 每次 `load()` 全量重读并重新解析 → 权限文件改动热生效。
- MCP 工具仅当配置权限为 `"read"` 且工具 `effect=="read"` 才放行；本章不对外部写工具放行。
- 内置工具（`query_order/query_product/query_faq/submit_refund/create_ticket`）不受该文件影响，
  始终注册，写工具另走服务端确认。
- 未知工具、未配置的 server 默认拒绝。MCP 自述的 `annotations` 一律视为不可信，不用于授权。

## 6. MySQL DDL 说明

工具审计表 `tool_audit_logs` 由 `sql/ch08-tool-audit.sql` 创建（临时验收库已应用）。要点：

- `id/conversation_id` 为 `BIGINT UNSIGNED`，`retry_count` 为 `TINYINT UNSIGNED`，
  `duration_ms` 为 `INT UNSIGNED`。
- `status` 为中文 ENUM：`('成功','失败','超时','校验拦下','权限拒绝')`；`tool_source` 为
  `ENUM('builtin','mcp')`。
- 不挂外键（审计写入不被引用约束拦住），只建普通索引；`ENGINE=InnoDB`、`CHARSET=utf8mb4`。
- 其余业务表（conversations/messages/tickets/orders 等）由 App 启动时
  `Base.metadata.create_all` 自动创建。

## 7. 证据摘要（验收口径）

- `tests/test_ch08_integration.py`：真实 adapters 双 Server 发现/调用；新工具授权前不可见 →
  只重启该 Server + 改权限文件后下一轮可见（App PID 不变）；真实模型轨迹含新公开名；
  真实 HTTP 确认/取消 + MySQL 落行；两独立进程并发争抢同一工单号只产生一行；DDL 属性核对。
  门控 `CH08_LIVE=1`（真实 LLM + MySQL 容器 + 本地 MCP 子进程），默认 `pytest -q` 保持 hermetic，
  门控未开时显式打印原因，不静默跳过。
- `scripts/demo_ch08.py`：五种模式的机器可读证据演示。
- `eval/eval_ch08_tools.py --mode trajectory`：真实模型 MCP 工具轨迹（双 Server + 热新增工具），
  逐例报告，模型不调用预期工具即 FAIL，绝不编造/改标。

## 8. 限制与未验证项

- 真实网络/模型路径依赖 `CH08_LIVE=1` 与临时 MySQL 容器 `knowweave-ch08-mysql`（127.0.0.1:3307）；
  未在 CI 里常态化执行。
- MCP 传输故障重试上限、超时预算为配置项（`tool_timeout`/`tool_retries`/`mcp_discovery_timeout`），
  不同部署需按实际时延调参。
- 前端（按钮/卡片 UI、表单交互）不在本章后端验收范围内。
- 未验证项（截至本文档）：大规模并发下的审计写入吞吐、跨 Server 同名工具的极端冲突场景
  （前缀隔离已覆盖，但未做压力验证）。如有变动以验收报告为准。

## 9. 留痕

阶段性原始留痕在本地文件 `dev-notes/ch08.md`（不入 Git，不随仓库分发）。
