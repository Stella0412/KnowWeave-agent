from dataclasses import replace

import pytest


def definition(name="query_product", **changes):
    from app.tools.contracts import ToolDefinition, ToolOutcome

    async def invoke(arguments, context):
        return ToolOutcome("success", data=arguments)

    return ToolDefinition(
        name=name, description="查询商品", parameters={"type": "object", "properties": {
            "count": {"type": "integer", "minimum": 1}}, "required": ["count"]},
        source="builtin", server=None, original_name=None, invoke=invoke, **changes,
    )


def test_registered_tool_is_visible_in_next_snapshot():
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    before = registry.snapshot()
    registry.register(definition())
    assert before == {}
    assert set(registry.snapshot()) == {"query_product"}


def test_duplicate_registration_does_not_replace_original():
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    tool = definition()
    registry.register(tool)
    with pytest.raises(ValueError, match="query_product"):
        registry.register(replace(tool, description="替换"))
    assert registry.snapshot()["query_product"].description == "查询商品"


def test_schema_mutation_cannot_change_registry_or_another_snapshot():
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    tool = definition()
    registry.register(tool)
    tool.parameters["properties"]["count"]["minimum"] = 100
    first = registry.snapshot()
    first["query_product"].parameters["properties"]["count"]["minimum"] = 200
    first.clear()
    assert registry.snapshot()["query_product"].parameters["properties"]["count"]["minimum"] == 1


def test_invalid_schema_is_rejected_without_registering_tool():
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register(replace(definition(), parameters={"type": "not-a-json-type"}))
    assert registry.snapshot() == {}


@pytest.mark.asyncio
async def test_async_handler_uses_only_current_execution_context():
    from app.tools.contracts import ExecutionContext, ToolOutcome
    from app.tools.registry import ToolRegistry

    async def invoke(arguments, context):
        return ToolOutcome("success", data={"conversation_id": context.conversation_id})

    registry = ToolRegistry()
    registry.register(replace(definition(), invoke=invoke))
    handler = registry.snapshot()["query_product"].invoke
    first = await handler({"count": 1}, ExecutionContext(10, "call-1"))
    second = await handler({"count": 1}, ExecutionContext(20, "call-2"))
    assert first.data == {"conversation_id": 10}
    assert second.data == {"conversation_id": 20}
