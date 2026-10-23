"""ch08 · Agent 动态工具调用测试（内置注册 + 每次工具调用单独节点推进）。"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import app.graph.agent as agent_module
from app.graph.agent import agent_llm, agent_tools
from app.graph.build import build_graph
from app.graph.state import GraphContext
from app.rag.pipeline import TurnContext
from app.services.token_counter import TokenCounter
from app.tools.builtin import build_tool_snapshot, register_builtins
from app.tools.contracts import ToolDefinition, ToolOutcome
from app.tools.registry import ToolRegistry


class FakeModel:
    """记录 bind_tools 收到的 dict Schema，按需返回消息。"""

    def __init__(self, message=None):
        self._message = message or AIMessage(content="答")
        self.bound = None
        self.calls = []

    def bind_tools(self, tools):
        self.bound = [t["name"] for t in tools]
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self._message


class _FakeTool:
    def __init__(self, name, *, description="", parameters=None, result=None):
        self.name = name
        self.description = description
        self.args_schema = parameters if parameters is not None else {"type": "object"}
        self.result = result
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return self.result


class _FakeClient:
    def __init__(self, servers):
        self.connections = {s: {} for s in servers}
        self._tools = {s: [] for s in servers}

    def set_tools(self, server, tools):
        self._tools[server] = list(tools)

    async def get_tools(self, *, server_name=None):
        return list(self._tools.get(server_name, []))


def _runtime(registry=None, provider=None, engine=None, model=None, session_factory=None):
    return SimpleNamespace(
        context=SimpleNamespace(
            model=model or FakeModel(),
            session_factory=session_factory,
            turn=TurnContext(raw_question="测试", conversation_id=1),
            token_budget=4096,
            tool_timeout=5.0,
            tool_retries=1,
            counter=TokenCounter(estimate_factor=1.0),
            tool_result_max_tokens=1200,
            context_manager=None,
            budget=None,
            settings=None,
            registry=registry,
            provider=provider,
            engine=engine,
        )
    )


def _state(**overrides):
    state = {
        "messages": [HumanMessage(content="问")],
        "conversation_id": 1,
        "route": "business",
        "intent": None,
        "steps": 0,
        "tokens_used": 0,
        "suggested_actions": [],
        "citations": [],
        "pending_tool_calls": [],
        "tool_cursor": 0,
        "tool_tokens_used": 0,
        "refund_offered": None,
        "trace": {},
    }
    state.update(overrides)
    return state


# 1. 运行中注册 demo_echo：下一次 bind_tools 收到其 Schema 且引擎执行成功；核心无 demo 分支

async def test_runtime_registered_tool_is_bound_and_executed():
    registry = ToolRegistry()
    register_builtins(registry)

    seen = []

    async def echo(arguments, ctx):
        seen.append(arguments)
        return ToolOutcome("success", data={"text": arguments["text"]})

    registry.register(ToolDefinition(
        name="demo_echo", description="回显测试文本", source="builtin",
        server=None, original_name="demo_echo", invoke=echo,
        parameters={"type": "object", "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
    ))

    model = FakeModel()
    runtime = _runtime(registry=registry, model=model)
    await agent_llm(_state(route="business"), runtime)
    assert "demo_echo" in model.bound

    state = _state(
        messages=[HumanMessage(content="回显"), AIMessage(content="", tool_calls=[{
            "name": "demo_echo", "args": {"text": "你好"}, "id": "c1", "type": "tool_call",
        }])],
        pending_tool_calls=[{
            "name": "demo_echo", "args": {"text": "你好"}, "id": "c1", "type": "tool_call",
        }],
        tool_cursor=0,
    )
    update = await agent_tools(state, runtime)
    message = update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "你好" in message.content
    assert seen == [{"text": "你好"}]
    assert update["tool_cursor"] == 1


# 2. 伪造/未注册工具名：错误 ToolMessage + 校验拦下审计，且不触发任何处理器

async def test_forged_tool_name_errors_and_audits(session_factory):
    registry = ToolRegistry()
    register_builtins(registry)
    state = _state(pending_tool_calls=[{
        "name": "forged_tool", "args": {}, "id": "c1", "type": "tool_call",
    }], tool_cursor=0)
    update = await agent_tools(state, _runtime(registry=registry, session_factory=session_factory))

    message = update["messages"][0]
    assert "未注册" in message.content
    assert "无法确定来源" in message.content

    from sqlalchemy import select

    from app.db.models import ToolAuditLog
    async with session_factory() as session:
        rows = (await session.execute(select(ToolAuditLog))).scalars().all()
    assert len(rows) == 1
    assert rows[0].tool_name == "forged_tool"
    assert rows[0].status.value == "校验拦下"


# 3. 已发现但未授权（撤权）的工具不执行，也不出现在下一快照

async def test_unauthorized_mcp_tool_not_executed(session_factory, tmp_path):
    from app.tools.mcp_client import MCPToolProvider
    from app.tools.policy import PolicyStore

    policy_path = tmp_path / "permissions.json"
    policy_path.write_text(json.dumps(
        {"mcp": {"after_sales": {"query_warranty": "read"}}}, ensure_ascii=False,
    ), encoding="utf-8")
    policy = PolicyStore(policy_path)
    client = _FakeClient(["logistics", "after_sales"])
    logistics_tool = _FakeTool("logistics_query_logistics")
    client.set_tools("logistics", [logistics_tool])
    provider = MCPToolProvider(client, policy, discovery_timeout=1.0)

    registry = ToolRegistry()
    register_builtins(registry)
    definitions = await build_tool_snapshot(registry, provider)
    assert "logistics_query_logistics" not in definitions

    state = _state(pending_tool_calls=[{
        "name": "logistics_query_logistics", "args": {"tracking_no": "SF1"},
        "id": "c1", "type": "tool_call",
    }], tool_cursor=0)
    update = await agent_tools(
        state, _runtime(registry=registry, provider=provider, session_factory=session_factory),
    )
    assert "未注册" in update["messages"][0].content
    assert logistics_tool.calls == []  # 处理器从未被调用


# 4. query_logistics 不在内置注册表快照里

def test_query_logistics_not_in_builtin_registry_snapshot():
    registry = ToolRegistry()
    register_builtins(registry)
    snapshot = registry.snapshot()
    assert "query_logistics" not in snapshot
    assert set(snapshot) == {
        "query_order", "query_product", "query_faq", "submit_refund", "create_ticket",
    }


# 5a. 一条 AIMessage 的多次工具调用逐节点执行，游标推进，每次只追加一条 ToolMessage

async def test_multiple_tool_calls_executed_one_per_node_invocation():
    registry = ToolRegistry()
    register_builtins(registry)
    executed = []

    async def probe(arguments, ctx):
        executed.append(arguments)
        return ToolOutcome("success", data=arguments)

    registry.register(ToolDefinition(
        name="probe", description="探针", source="builtin", server=None,
        original_name="probe", invoke=probe,
        parameters={"type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"]},
    ))

    runtime = _runtime(registry=registry)
    pending = [
        {"name": "probe", "args": {"n": 1}, "id": "c1", "type": "tool_call"},
        {"name": "probe", "args": {"n": 2}, "id": "c2", "type": "tool_call"},
    ]
    state = _state(pending_tool_calls=pending, tool_cursor=0)

    first = await agent_tools(state, runtime)
    assert len(first["messages"]) == 1
    assert first["tool_cursor"] == 1
    assert executed == [{"n": 1}]

    state["tool_cursor"] = first["tool_cursor"]
    state["tool_tokens_used"] = first["tool_tokens_used"]
    state["messages"] = state["messages"] + first["messages"]
    second = await agent_tools(state, runtime)
    assert len(second["messages"]) == 1
    assert second["tool_cursor"] == 2
    assert executed == [{"n": 1}, {"n": 2}]


# 5b. 后续调用触发 interrupt 时，已完成的调用不被重放（checkpoint 推进）

async def test_interrupt_does_not_replay_completed_calls(session_factory):
    from langgraph.types import Command, interrupt

    class InterruptOnceEngine:
        def __init__(self):
            self.c1_runs = 0
            self.c2_runs = 0
            self.interrupted = False

        async def execute(self, call, context, definitions):
            call_id = call.get("id")
            if call_id == "c1":
                self.c1_runs += 1
            else:
                self.c2_runs += 1
            if call_id == "c2" and not self.interrupted:
                self.interrupted = True
                interrupt("pause before c2")
            return ToolMessage(
                content="ok", tool_call_id=call_id, name=call.get("name"),
            )

    engine = InterruptOnceEngine()
    model = _ScriptedAgentModel([
        AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"},
            {"name": "query_order", "args": {"order_id": "1002"}, "id": "c2", "type": "tool_call"},
        ]),
        AIMessage(content="已查到"),
    ])
    graph = build_graph(checkpointer=InMemorySaver())
    context = _graph_context(session_factory, model, engine)

    result = await graph.ainvoke(
        _graph_input(), config={"configurable": {"thread_id": "t1"}}, context=context,
    )
    assert "__interrupt__" in result
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["c1"]
    assert engine.c1_runs == 1  # c1 完成一次即 checkpoint，未被中断前的循环重放

    resumed = await graph.ainvoke(
        Command(resume="继续"),
        config={"configurable": {"thread_id": "t1"}},
        context=context,
    )
    tool_messages = [m for m in resumed["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    assert engine.c1_runs == 1  # 恢复后 c1 没有被重放
    assert engine.c2_runs == 2  # c2 第一次被中断，恢复后重试成功


# 6. submit_refund 只有订单一致且符合资格才产出表单；拒绝无动作且审计权限拒绝

async def test_submit_refund_eligible_produces_form(session_factory):
    registry = ToolRegistry()
    register_builtins(registry)
    signed = (datetime.now() - timedelta(days=3)).isoformat()
    state = _state(
        route="refund_flow",
        intent="refund_return",
        order_no="1001",
        order_data={"order_no": "1001", "product_name": "猫爬架", "status": "已签收",
                    "product_type": "普通", "signed_at": signed},
        pending_tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "1001"},
            "id": "c1", "type": "tool_call",
        }],
        tool_cursor=0,
    )
    update = await agent_tools(state, _runtime(registry=registry, session_factory=session_factory))
    action = update["suggested_actions"][0]
    assert action["type"] == "refund_form"
    assert action["draft"]["order_no"] == "1001"
    assert update["refund_offered"] is True


async def test_submit_refund_ineligible_denied_and_audited(session_factory):
    registry = ToolRegistry()
    register_builtins(registry)
    signed = (datetime.now() - timedelta(days=3)).isoformat()
    state = _state(
        route="refund_flow",
        intent="refund_return",
        order_no="1002",
        order_data={"order_no": "1002", "product_name": "定制猫窝", "status": "已签收",
                    "product_type": "定制", "signed_at": signed},
        pending_tool_calls=[{
            "name": "submit_refund", "args": {"order_no": "1002"},
            "id": "c1", "type": "tool_call",
        }],
        tool_cursor=0,
    )
    update = await agent_tools(state, _runtime(registry=registry, session_factory=session_factory))
    assert update["suggested_actions"] == []
    assert update["refund_offered"] is False
    assert "定制" in update["messages"][0].content

    from sqlalchemy import select

    from app.db.models import ToolAuditLog
    async with session_factory() as session:
        rows = (await session.execute(select(ToolAuditLog))).scalars().all()
    assert len(rows) == 1
    assert rows[0].status.value == "权限拒绝"


# —— 图级测试夹具 ——


class _FakeIntentModel:
    def __init__(self, intent):
        self.intent = intent

    async def ainvoke(self, messages):
        from app.prompts.orchestrator import IntentOutput
        return IntentOutput(intent=self.intent, confidence=0.9)


class _ScriptedAgentModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.bound = None

    def bind_tools(self, tools):
        self.bound = [t["name"] for t in tools]
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return self.responses.pop(0)


def _graph_context(session_factory, model, engine):
    return GraphContext(
        model=model,
        intent_model=_FakeIntentModel("order"),
        session_factory=session_factory,
        turn=TurnContext(raw_question="查订单", conversation_id=1),
        engine=engine,
    )


def _graph_input():
    return {
        "messages": [HumanMessage(content="查订单")],
        "user_id": "anonymous",
        "conversation_id": 1,
        "context_meta": {},
        "intent": None,
        "intent_source": None,
        "route": None,
        "evidence": None,
        "citations": [],
        "evidence_strong": None,
        "retrieval_error": None,
        "refusal_source": None,
        "refusal_reason": None,
        "answer": None,
        "steps": 0,
        "tokens_used": 0,
        "suggested_actions": [],
        "fallback_reason": None,
        "stop_reason": None,
        "trace": {"t0": 0.0},
    }
