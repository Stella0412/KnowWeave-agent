"""ch08 即插即用工具系统的注册契约。

后续任务（执行引擎、审计、MCP 发现、Agent 集成）都依赖这些 dataclass 的字段名与顺序，
不得重命名或调整位置。
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class ToolDefinition:
    """统一工具描述：内置与 MCP 工具共用。

    字段顺序固定（测试按位置构造）：
    name / description / parameters / source / server / original_name / invoke，
    其后 effect / formatter / guard 带默认值。
    """

    name: str
    description: str
    parameters: dict
    source: str
    server: str | None
    original_name: str | None
    invoke: Callable[..., Awaitable[Any]]
    effect: str = "read"
    formatter: Callable | None = None
    guard: Callable | None = None


@dataclass
class ExecutionContext:
    """一次工具调用的可信上下文；模型参数不得注入这里。"""

    conversation_id: int
    tool_call_id: str
    session_factory: Any = None
    turn: Any = None
    ticket_request: Any = None
    confirmation: Any = None
    state: Any = None
    ticket_decision: str | None = None
    confirmation_reason: str | None = None


@dataclass
class ToolOutcome:
    """统一执行结果。"""

    category: str
    data: Any = None
    message: str | None = None
    actions: tuple = ()


@dataclass
class AuditRecord:
    """一次逻辑工具调用的审计终态。"""

    conversation_id: int | None
    tool_call_id: str
    tool_name: str
    tool_source: str
    mcp_server: str | None
    arguments: Any
    result_summary: str | None
    status: str
    error_message: str | None
    retry_count: int
    duration_ms: int


class TransientToolError(Exception):
    """暂时性工具故障（限流、上游临时不可用等），读工具可安全重试。"""
