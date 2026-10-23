"""ch08 · 统一执行引擎（ToolEngine）行为测试。

夹具全部本地定义：engine timeout=0.01、retries=2，audit_writer 向列表追加，
注入 fake sleep 记录退避等待。写工具的确认通过 ExecutionContext.confirmation 提供，
绝不塞进模型 args。
"""

import asyncio
import json

import pytest

from app.tools.contracts import ExecutionContext, ToolDefinition, ToolOutcome, TransientToolError


class _Snapshot:
    def __init__(self, permit):
        self._permit = permit

    def permits(self, definition):
        return self._permit


class _Policy:
    def __init__(self, permit=True):
        self._permit = permit
        self.loads = 0

    def load(self):
        self.loads += 1
        return _Snapshot(self._permit)


async def _confirm_true(definition, arguments, context):
    return True


async def _confirm_false(definition, arguments, context):
    return False


@pytest.fixture
def audits():
    return []


@pytest.fixture
def slept():
    return []


@pytest.fixture
def engine(audits, slept):
    from app.tools.executor import ToolEngine

    async def fake_sleep(seconds):
        slept.append(seconds)

    async def writer(record):
        audits.append(record)

    return ToolEngine(policy=_Policy(permit=True), audit_writer=writer,
                      timeout=0.01, retries=2, sleep=fake_sleep)


@pytest.fixture
def register_tool():
    definitions = {}

    def _register(name, invoke, *, effect="read", formatter=None, source="builtin",
                  server=None, original_name=None, parameters=None):
        params = parameters if parameters is not None else {"type": "object"}
        definitions[name] = ToolDefinition(
            name=name, description="", parameters=params, source=source,
            server=server, original_name=original_name, invoke=invoke,
            effect=effect, formatter=formatter,
        )
        return definitions

    return _register


@pytest.fixture
def context():
    return ExecutionContext(conversation_id=1, tool_call_id="call-1")


async def test_invalid_arguments_block_handler_and_audit(engine, register_tool, context, audits):
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        return ToolOutcome("success")

    definitions = register_tool("t", handler, parameters={
        "type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]})

    message = await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)

    assert calls == []
    record = audits[-1]
    assert record.status == "校验拦下"
    assert record.retry_count == 0
    parsed = json.loads(message.content)
    assert parsed["category"] == "invalid_arguments"
    assert parsed["ok"] is False


async def test_permission_denied_blocks_handler(register_tool, context, audits):
    from app.tools.executor import ToolEngine

    async def writer(record):
        audits.append(record)

    async def fake_sleep(seconds):
        pass

    engine = ToolEngine(policy=_Policy(permit=False), audit_writer=writer,
                        timeout=0.01, retries=2, sleep=fake_sleep)
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        return ToolOutcome("success")

    definitions = register_tool("t", handler)
    message = await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)

    assert calls == []
    assert audits[-1].status == "权限拒绝"
    assert json.loads(message.content)["category"] == "permission_denied"


async def test_unknown_tool_audits_validation_blocked(engine, context, audits):
    message = await engine.execute({"id": "c1", "name": "missing", "args": {}}, context, {})

    record = audits[-1]
    assert record.status == "校验拦下"
    assert record.tool_source == "builtin"
    assert record.mcp_server is None
    assert "未注册，无法确定来源" in record.error_message
    assert json.loads(message.content)["category"] == "invalid_arguments"


async def test_read_transient_then_success(engine, register_tool, context, audits):
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        if len(calls) == 1:
            raise TransientToolError("上游暂时不可用")
        return ToolOutcome("success", data={"ok": 1})

    definitions = register_tool("t", handler)
    message = await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)

    assert len(calls) == 2
    record = audits[-1]
    assert record.retry_count == 1
    assert record.status == "成功"
    assert json.loads(message.content)["ok"] is True


async def test_read_always_times_out(engine, register_tool, context, audits, slept):
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        await asyncio.Event().wait()

    definitions = register_tool("t", handler)
    message = await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)

    assert len(calls) == 3  # retries(2) + 1
    record = audits[-1]
    assert record.status == "超时"
    assert record.retry_count == 2
    assert slept  # 退避等待确实被注入的 sleep 执行
    assert json.loads(message.content)["category"] == "fault"


async def test_read_not_found_no_retry(engine, register_tool, context, audits):
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        return ToolOutcome("not_found", data=None, message="未查询到记录")

    definitions = register_tool("t", handler)
    message = await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)

    assert len(calls) == 1
    assert audits[-1].status == "成功"
    parsed = json.loads(message.content)
    assert parsed["ok"] is True
    assert parsed["category"] == "not_found"


async def test_read_permanent_error(engine, register_tool, context, audits):
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        raise ValueError("参数无效")

    definitions = register_tool("t", handler)
    message = await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)

    assert len(calls) == 1
    assert audits[-1].status == "失败"
    assert json.loads(message.content)["category"] == "fault"


async def test_write_timeout_never_retries(engine, register_tool, audits):
    calls = 0

    async def slow(arguments, ctx):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    definitions = register_tool("create_ticket", slow, effect="write")
    ctx = ExecutionContext(1, "w1", confirmation=_confirm_true)
    message = await engine.execute({"id": "w1", "name": "create_ticket", "args": {}}, ctx, definitions)

    assert calls == 1
    assert audits[-1].status == "超时"
    assert audits[-1].retry_count == 0
    assert audits[-1].duration_ms >= 0
    assert "执行状态可能不确定" in json.loads(message.content)["message"]


async def test_write_missing_confirmation_denied(engine, register_tool, context, audits):
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        return ToolOutcome("success")

    definitions = register_tool("create_ticket", handler, effect="write")
    message = await engine.execute({"id": "w1", "name": "create_ticket", "args": {}},
                                   context, definitions)

    assert calls == []
    assert audits[-1].status == "权限拒绝"
    assert json.loads(message.content)["category"] == "permission_denied"


async def test_write_confirmation_true_executes(engine, register_tool, audits):
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        return ToolOutcome("success", data={"ticket_no": "T-1"})

    definitions = register_tool("create_ticket", handler, effect="write")
    ctx = ExecutionContext(1, "w1", confirmation=_confirm_true)
    message = await engine.execute({"id": "w1", "name": "create_ticket", "args": {}},
                                   ctx, definitions)

    assert len(calls) == 1
    assert audits[-1].status == "成功"
    assert json.loads(message.content)["data"]["ticket_no"] == "T-1"


async def test_write_confirmation_false_denied(engine, register_tool, audits):
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        return ToolOutcome("success")

    definitions = register_tool("create_ticket", handler, effect="write")
    ctx = ExecutionContext(1, "w1", confirmation=_confirm_false)
    message = await engine.execute({"id": "w1", "name": "create_ticket", "args": {}},
                                   ctx, definitions)

    assert calls == []
    assert audits[-1].status == "权限拒绝"
    assert json.loads(message.content)["category"] == "permission_denied"


async def test_write_transient_zero_retries(engine, register_tool, audits):
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        raise TransientToolError("暂时失败")

    definitions = register_tool("create_ticket", handler, effect="write")
    ctx = ExecutionContext(1, "w1", confirmation=_confirm_true)
    message = await engine.execute({"id": "w1", "name": "create_ticket", "args": {}},
                                   ctx, definitions)

    assert len(calls) == 1
    assert audits[-1].retry_count == 0
    assert audits[-1].status == "失败"


async def test_audit_writer_failure_isolated(register_tool, context):
    from app.tools.executor import ToolEngine

    async def failing_writer(record):
        raise RuntimeError("audit down")

    async def fake_sleep(seconds):
        pass

    engine = ToolEngine(policy=_Policy(permit=True), audit_writer=failing_writer,
                        timeout=0.01, retries=2, sleep=fake_sleep)

    async def handler(arguments, ctx):
        return ToolOutcome("success", data={"x": "y"})

    definitions = register_tool("t", handler)
    message = await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)

    parsed = json.loads(message.content)
    assert parsed["ok"] is True
    assert parsed["category"] == "success"
    assert parsed["data"] == {"x": "y"}


async def test_cancelled_error_propagates(engine, register_tool, context):
    async def handler(arguments, ctx):
        raise asyncio.CancelledError()

    definitions = register_tool("t", handler)
    with pytest.raises(asyncio.CancelledError):
        await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)


async def test_chinese_data_not_ascii_escaped(engine, register_tool, context):
    async def handler(arguments, ctx):
        return ToolOutcome("success", data={"承运商": "顺丰", "状态": "在途"})

    definitions = register_tool("t", handler)
    message = await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)

    assert "顺丰" in message.content
    assert "\\u" not in message.content
    assert json.loads(message.content)["data"]["承运商"] == "顺丰"


async def test_formatter_filters_and_translates(engine, register_tool, context):
    def fmt(outcome):
        data = {k: v for k, v in (outcome.data or {}).items() if k != "secret"}
        if data.get("status") == "delivered":
            data["status"] = "已签收"
        return ToolOutcome(outcome.category, data=data, message=outcome.message)

    async def handler(arguments, ctx):
        return ToolOutcome("success", data={"secret": "x", "status": "delivered"})

    definitions = register_tool("t", handler, formatter=fmt)
    message = await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)

    parsed = json.loads(message.content)
    assert "secret" not in parsed["data"]
    assert parsed["data"]["status"] == "已签收"


async def test_formatter_raising_yields_fault(engine, register_tool, context):
    def bad_fmt(outcome):
        raise RuntimeError("格式化崩溃")

    async def handler(arguments, ctx):
        return ToolOutcome("success", data={"x": 1})

    definitions = register_tool("t", handler, formatter=bad_fmt)
    message = await engine.execute({"id": "c1", "name": "t", "args": {}}, context, definitions)

    parsed = json.loads(message.content)
    assert parsed["category"] == "fault"
    assert parsed["ok"] is False


async def test_missing_args_treated_as_empty_object(engine, register_tool, context):
    calls = []

    async def handler(arguments, ctx):
        calls.append(arguments)
        return ToolOutcome("success")

    definitions = register_tool("t", handler)
    await engine.execute({"id": "c1", "name": "t"}, context, definitions)

    assert calls == [{}]


async def test_non_object_args_fail_validation(engine, register_tool, context, audits):
    calls = []

    async def handler(arguments, ctx):
        calls.append(1)
        return ToolOutcome("success")

    definitions = register_tool("t", handler)
    message = await engine.execute({"id": "c1", "name": "t", "args": [1, 2]},
                                   context, definitions)

    assert calls == []
    assert audits[-1].status == "校验拦下"
    assert json.loads(message.content)["category"] == "invalid_arguments"
