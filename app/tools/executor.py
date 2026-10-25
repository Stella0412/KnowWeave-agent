import asyncio
import inspect
import logging
import time

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from app.tools.contracts import AuditRecord, ToolOutcome, TransientToolError
from app.tools.formatting import format_outcome, summarize_for_audit
from app.tools.validation import argument_errors
from app.observability.spans import _summarize_args

logger = logging.getLogger(__name__)

_MAX_ERROR_MESSAGE = 512
_BACKOFF_BASE = 0.05
_BACKOFF_MAX = 1.0


def _format_argument_errors(errors: list[dict]) -> str:
    parts = []
    for e in errors:
        path = "/".join(str(p) for p in e["path"]) or "(root)"
        message = e["message"]
        if "allowed_values" in e:
            message += f"（允许值: {', '.join(map(str, e['allowed_values']))}）"
        parts.append(f"{path}: {message}")
    return "参数校验失败: " + "; ".join(parts)


def _truncate_error(message: str | None) -> str | None:
    if message is None:
        return None
    return message if len(message) <= _MAX_ERROR_MESSAGE else message[:_MAX_ERROR_MESSAGE]


class ToolEngine:
    """ch08 统一执行引擎：校验 → 权限 → 写确认 → 执行/重试 → 格式化 → 审计 → ToolMessage。

    构造参数：policy（含 load() 返回带 permits() 的快照）、audit_writer（async，接收
    AuditRecord）、timeout、retries、sleep（注入退避等待，默认 asyncio.sleep）。
    """

    def __init__(self, policy, audit_writer, timeout, retries, sleep=asyncio.sleep, tracer=None) -> None:
        self.policy = policy
        self.audit_writer = audit_writer
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep
        self.tracer = tracer

    def _resolve_timeout(self, name: str) -> float:
        """timeout 既可以是标量，也可以是按工具名给预算的 callable。"""
        if callable(self.timeout):
            return self.timeout(name)
        return self.timeout

    async def execute(self, tool_call, context, tools_by_name):
        name = tool_call.get("name", "")
        args = tool_call.get("args") or {}
        if self.tracer is None:
            return await self._execute(tool_call, context, tools_by_name)
        with self.tracer(
            name, input=_summarize_args(args),
            metadata={"tool_call_id": context.tool_call_id,
                      "turn_token": getattr(context, "turn_token", None),
                      "retry": 0},
        ) as span:
            message = await self._execute(tool_call, context, tools_by_name)
            if span is not None:
                category = (getattr(message, "artifact", None) or {}).get("category")
                span.update(
                    output=str(message.content)[:500],
                    level="DEFAULT" if category == "success" else "ERROR",
                    status_message=None if category == "success" else str(message.content)[:500],
                )
            return message

    async def _execute(self, tool_call: dict, context, tools_by_name: dict) -> ToolMessage:
        name = tool_call.get("name", "")
        call_id = tool_call.get("id", "")
        args = tool_call.get("args")
        if args is None:
            args = {}

        definition = tools_by_name.get(name)

        # 1. 解析工具身份：未知名称产生参数类错误，不猜测来源
        if definition is None:
            message = f"未知工具: {name}，未注册，无法确定来源"
            return await self._emit(
                context, name, call_id, args, None,
                ToolOutcome("invalid_arguments", data=None, message=message),
                status="校验拦下", error_message=message, retry_count=0, duration_ms=0,
            )

        # 2. JSON Schema 校验
        errors = argument_errors(definition.parameters, args)
        if errors:
            error_message = _format_argument_errors(errors)
            return await self._emit(
                context, name, call_id, args, definition,
                ToolOutcome("invalid_arguments", data={"errors": errors}, message="参数校验失败"),
                status="校验拦下", error_message=error_message, retry_count=0, duration_ms=0,
            )

        # 3. 每次执行重新加载权限快照，热更新立即生效
        snapshot = self.policy.load()
        if not snapshot.permits(definition):
            message = f"工具权限被拒绝: {name}"
            return await self._emit(
                context, name, call_id, args, definition,
                ToolOutcome("permission_denied", data=None, message=message),
                status="权限拒绝", error_message=message, retry_count=0, duration_ms=0,
            )

        # 3.5 可信本地守卫（在权限放行之后、确认/执行之前应用）
        if definition.guard is not None:
            try:
                result = definition.guard(args, context)
                if inspect.isawaitable(result):
                    result = await result
            except GraphBubbleUp:
                raise
            except Exception as e:  # 守卫崩溃即 fail-closed，不执行；不捕 BaseException
                message = f"工具守卫失败: {e}"
                return await self._emit(
                    context, name, call_id, args, definition,
                    ToolOutcome("fault", data=None, message=message),
                    status="失败", error_message=str(e), retry_count=0, duration_ms=0,
                )
            if result is not None:
                # 守卫返回 ToolOutcome 即拦截：不执行，记权限拒绝，返回格式化后的 ToolMessage
                return await self._emit(
                    context, name, call_id, args, definition, result,
                    status="权限拒绝", error_message=result.message, retry_count=0, duration_ms=0,
                )

        # 4. 写工具需要可信确认入口；确认失败不执行
        if definition.effect == "write":
            confirmation = context.confirmation
            if confirmation is None:
                message = f"写工具 {name} 缺少确认入口，已拒绝执行"
                return await self._emit(
                    context, name, call_id, args, definition,
                    ToolOutcome("permission_denied", data=None, message=message),
                    status="权限拒绝", error_message=message, retry_count=0, duration_ms=0,
                )
            try:
                confirmed = await confirmation(definition, args, context)
            except GraphBubbleUp:
                raise
            except Exception as e:  # 不捕 BaseException，取消/中断向上传播
                message = f"写操作确认阶段失败: {e}"
                return await self._emit(
                    context, name, call_id, args, definition,
                    ToolOutcome("fault", data=None, message=message),
                    status="失败", error_message=str(e), retry_count=0, duration_ms=0,
                )
            if not confirmed:
                reason = getattr(context, "confirmation_reason", None)
                message = reason if reason else f"写操作未获用户确认: {name}"
                return await self._emit(
                    context, name, call_id, args, definition,
                    ToolOutcome("permission_denied", data=None, message=message),
                    status="权限拒绝", error_message=message, retry_count=0, duration_ms=0,
                )

        # 5. 执行（写工具从确认返回 True 后才开始计时，不含用户确认等待）
        is_write = definition.effect == "write"
        start = time.perf_counter()
        outcome, status, error_message, retry_count = await self._run(
            definition, args, context, retry_allowed=not is_write, is_write=is_write,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)

        # 6. 注册格式化器：字段筛选 / 枚举翻译；格式化器崩溃不拖垮引擎
        outcome, formatter_failed = self._apply_formatter(definition, outcome)
        if formatter_failed:
            status = "失败"
            error_message = outcome.message

        # 7. 格式化 → 尽力审计 → ToolMessage
        return await self._emit(
            context, name, call_id, args, definition, outcome,
            status=status, error_message=error_message, retry_count=retry_count,
            duration_ms=duration_ms,
        )

    async def _run(self, definition, args, context, *, retry_allowed: bool, is_write: bool):
        """执行处理器并按故障类别分类。返回 (outcome, status, error_message, retry_count)。"""
        result = None
        succeeded = False
        last_error: Exception | None = None
        retry_count = 0

        max_attempts = self.retries + 1 if retry_allowed else 1

        for attempt in range(max_attempts):
            try:
                result = await asyncio.wait_for(
                    definition.invoke(args, context),
                    timeout=self._resolve_timeout(definition.name),
                )
                succeeded = True
                break
            except GraphBubbleUp:
                raise
            except (TimeoutError, asyncio.TimeoutError) as e:
                last_error = e
                if retry_allowed and attempt < self.retries:
                    retry_count += 1
                    await self.sleep(self._backoff(retry_count))
            except TransientToolError as e:
                last_error = e
                if retry_allowed and attempt < self.retries:
                    retry_count += 1
                    await self.sleep(self._backoff(retry_count))
            except Exception as e:  # 永久异常不重试；不捕 BaseException
                last_error = e
                break

        if succeeded:
            outcome = result if isinstance(result, ToolOutcome) else ToolOutcome("success", data=result)
            if outcome.category == "fault":
                # 处理器正常返回的 fault 是业务失败而非异常：记失败、不重试。
                return outcome, "失败", outcome.message or "工具返回失败", retry_count
            return outcome, "成功", None, retry_count

        if isinstance(last_error, (TimeoutError, asyncio.TimeoutError)):
            if is_write:
                message = "工具执行超时，执行状态可能不确定（可能已实际执行）"
            else:
                message = "工具执行超时"
            return ToolOutcome("fault", data=None, message=message), "超时", message, retry_count

        return ToolOutcome("fault", data=None, message=f"工具执行失败: {last_error}"), \
            "失败", str(last_error), retry_count

    def _apply_formatter(self, definition, outcome):
        if definition.formatter is None:
            return outcome, False
        try:
            return definition.formatter(outcome), False
        except Exception as e:
            return ToolOutcome("fault", data=None, message=f"结果格式化器失败: {e}"), True

    def _backoff(self, retry_count: int) -> float:
        return min(_BACKOFF_BASE * (2 ** retry_count), _BACKOFF_MAX)

    async def _emit(self, context, name, call_id, args, definition, outcome, *,
                    status, error_message, retry_count, duration_ms):
        content = format_outcome(outcome)
        record = AuditRecord(
            conversation_id=context.conversation_id,
            tool_call_id=context.tool_call_id,
            turn_token=getattr(context, "turn_token", None),
            tool_name=name,
            tool_source=definition.source if definition is not None else "builtin",
            mcp_server=definition.server if definition is not None else None,
            arguments=args,
            result_summary=summarize_for_audit(content),
            status=status,
            error_message=_truncate_error(error_message),
            retry_count=retry_count,
            duration_ms=duration_ms,
        )
        try:
            await self.audit_writer(record)
        except Exception:  # 审计失败隔离：不改结果、不把成功变失败
            logger.warning("audit_writer failed for tool %s", name, exc_info=True)
        return ToolMessage(
            content=content,
            tool_call_id=call_id,
            name=name,
            artifact={"category": outcome.category, "actions": list(outcome.actions)},
        )
