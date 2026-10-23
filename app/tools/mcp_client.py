"""ch08 · 通过官方 langchain-mcp-adapters 做 MCP 动态发现与本地授权适配。

- create_mcp_client(settings)：用两个 Streamable HTTP 端点构建 MultiServerMCPClient，
  tool_name_prefix=True、handle_tool_errors=False（MCP isError 走 ToolException 而非吞掉）。
- MCPToolProvider(client, policy, discovery_timeout)：refresh() 逐 server 限时发现，
  独立失败互不影响；发现出的工具转成 ToolDefinition，按本地权限快照过滤，未授权不暴露；
  MCP 描述/annotations 一律视为不可信，effect 固定 "read"。

结果适配：成功 → ToolOutcome("success", data=...)（优先结构化内容/JSON 解析）；
MCP isError（ToolException）→ ToolOutcome("fault")（确定性业务错误，不重试）；
传输层故障（超时 / httpx / 连接 / OSError）→ TransientToolError（读工具可重试）。
"""

import asyncio
import json
import logging

import httpx
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools.contracts import ToolDefinition, ToolOutcome, TransientToolError
from app.tools.validation import check_schema

logger = logging.getLogger(__name__)

# 可信格式化器：只翻译三个已知工具的稳定 ASCII 状态码，不虚构未知枚举含义。
_LOGISTICS_STATUS = {"pending_pickup": "待揽收", "in_transit": "运输中", "delivered": "已签收"}
_WARRANTY_STATUS = {"in_warranty": "在保", "out_of_warranty": "过保"}
_RETURN_STATUS = {"pending": "待处理", "in_transit": "退回中", "refunded": "已退款"}

# 传输层故障视为暂时性，读工具可重试。ConnectionError 是 OSError 子类，仍显式列出。
_TRANSIENT_EXCEPTIONS = (TimeoutError, asyncio.TimeoutError, httpx.HTTPError,
                         ConnectionError, OSError)


def create_mcp_client(settings) -> MultiServerMCPClient:
    """按 settings 构建两个 MCP Server 的连接；官方客户端、Streamable HTTP。"""
    connections = {
        "logistics": {"url": settings.mcp_logistics_url, "transport": "streamable_http"},
        "after_sales": {"url": settings.mcp_after_sales_url, "transport": "streamable_http"},
    }
    return MultiServerMCPClient(
        connections, tool_name_prefix=True, handle_tool_errors=False,
    )


def _extract_parameters(tool) -> dict | None:
    """取参数 JSON Schema：适配器传入 dict（MCP inputSchema），兼容 Pydantic 模型。"""
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return None
    if isinstance(schema, dict):
        return dict(schema)
    return schema.model_json_schema()


def _unwrap_result(raw):
    """把 MCP 调用结果规整为 ToolOutcome.data。

    优先结构化内容：ToolMessage.artifact（若有）/ dict / list 结果；JSON 字符串解析；
    否则保留文本。列表按 content block 取 text（真实服务器返回的 shape）。
    """
    if isinstance(raw, ToolMessage):
        artifact = getattr(raw, "artifact", None)
        if artifact is not None:
            if isinstance(artifact, dict) and set(artifact) == {"structured_content"}:
                return artifact["structured_content"]
            return artifact
        raw = raw.content

    if isinstance(raw, list) and raw and all(
        isinstance(block, dict) and "type" in block for block in raw
    ):
        texts = [block["text"] for block in raw
                 if block.get("type") == "text" and isinstance(block.get("text"), str)]
        if texts:
            raw = "\n".join(texts)
        else:
            return raw  # 图片/文件等非文本块，保留原样

    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except (ValueError, TypeError):
                pass
        return raw
    return raw


def _translate_status(data: dict, mapping: dict, entries_key: str | None) -> dict:
    """保留原始 ASCII status_code，追加中文 status_text；逐条翻译 events/progress。"""
    data = dict(data)
    code = data.get("status_code")
    if isinstance(code, str) and code in mapping:
        data["status_text"] = mapping[code]
    if entries_key is not None and isinstance(data.get(entries_key), list):
        translated = []
        for entry in data[entries_key]:
            if isinstance(entry, dict) and isinstance(entry.get("status_code"), str) \
                    and entry["status_code"] in mapping:
                entry = {**entry, "status_text": mapping[entry["status_code"]]}
            translated.append(entry)
        data[entries_key] = translated
    return data


def _make_formatter(mapping: dict, entries_key: str | None = None):
    def fmt(outcome):
        if outcome.category != "success" or not isinstance(outcome.data, dict):
            return outcome
        return ToolOutcome(outcome.category,
                           data=_translate_status(outcome.data, mapping, entries_key),
                           message=outcome.message, actions=outcome.actions)
    return fmt


_FORMATTERS = {
    ("logistics", "query_logistics"): _make_formatter(_LOGISTICS_STATUS, "events"),
    ("after_sales", "query_warranty"): _make_formatter(_WARRANTY_STATUS),
    ("after_sales", "query_return_progress"): _make_formatter(_RETURN_STATUS, "progress"),
}


class MCPToolProvider:
    """逐 server 限时发现 MCP 工具，转换为 ToolDefinition 并经本地权限过滤。"""

    def __init__(self, client, policy, discovery_timeout) -> None:
        self.client = client
        self.policy = policy
        self.discovery_timeout = discovery_timeout

    async def refresh(self) -> dict[str, ToolDefinition]:
        """重新发现所有 server 的工具，返回完整快照（键为公开名）。

        每次 refresh 都重新加载权限快照；某 server 失败/超时只记 warning，本轮不暴露
        该 server 的任何工具（包括上一轮已发现的），其余 server 继续。
        """
        tools: dict[str, ToolDefinition] = {}
        snapshot = self.policy.load()
        for server in list(self.client.connections):
            try:
                discovered = await asyncio.wait_for(
                    self.client.get_tools(server_name=server),
                    timeout=self.discovery_timeout,
                )
            except Exception as exc:  # 单个 server 故障不影响其他 server
                logger.warning("MCP 工具发现失败（server=%s）: %s", server, exc)
                continue
            for tool in discovered:
                definition = self._build(server, tool, snapshot)
                if definition is not None:
                    tools[definition.name] = definition
        return tools

    def _build(self, server, tool, snapshot) -> ToolDefinition | None:
        prefix = f"{server}_"
        if not tool.name.startswith(prefix):
            logger.warning("跳过未知前缀的 MCP 工具 %r（server=%s），不猜测来源",
                           tool.name, server)
            return None
        raw_name = tool.name[len(prefix):]

        parameters = _extract_parameters(tool)
        if parameters is None:
            logger.warning("MCP 工具 %s 无参数 Schema，跳过", tool.name)
            return None
        try:
            check_schema(parameters)  # 非法 Schema 注册中心也会拒绝，这里提前跳过
        except Exception as exc:
            logger.warning("MCP 工具 %s 参数 Schema 非法，跳过: %s", tool.name, exc)
            return None

        definition = ToolDefinition(
            name=tool.name,  # 公开名（server_raw_name）
            description=getattr(tool, "description", "") or "",
            parameters=parameters,
            source="mcp",
            server=server,
            original_name=raw_name,
            invoke=_make_invoke(tool),
            effect="read",  # 从不信任 MCP annotations
            formatter=_FORMATTERS.get((server, raw_name)),
        )

        if not snapshot.permits(definition):
            logger.debug("MCP 工具 %s 未被本地授权，不暴露", tool.name)
            return None
        return definition


def _make_invoke(tool):
    async def invoke(arguments, context):
        try:
            raw = await tool.ainvoke(arguments)
        except ToolException as exc:
            # MCP isError=True：确定性业务错误，不重试，也不把错误正文当成功数据
            return ToolOutcome("fault", data=None, message=str(exc))
        except _TRANSIENT_EXCEPTIONS as exc:
            raise TransientToolError(f"MCP 传输故障: {exc}") from exc
        return ToolOutcome("success", data=_unwrap_result(raw))
    return invoke
