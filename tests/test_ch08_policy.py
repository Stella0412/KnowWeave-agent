import json
from dataclasses import replace
from pathlib import Path

import pytest


def mcp_tool(name="public_alias", server="logistics", original_name="query_logistics", effect="read"):
    from app.tools.contracts import ToolDefinition, ToolOutcome

    async def invoke(arguments, context):
        return ToolOutcome("success", data=arguments)

    return ToolDefinition(name, "查询物流", {"type": "object"}, "mcp", server,
                          original_name, invoke, effect=effect)


def write_policy(path, tools):
    path.write_text(json.dumps({"mcp": {"logistics": tools}}, ensure_ascii=False), encoding="utf-8")


def test_unknown_mcp_tool_is_denied_by_default(tmp_path):
    from app.tools.policy import PolicyStore

    path = tmp_path / "permissions.json"
    write_policy(path, {"query_logistics": "read"})
    snapshot = PolicyStore(path).load()
    assert snapshot.permits(mcp_tool())
    assert not snapshot.permits(mcp_tool(original_name="new_tool"))
    assert not snapshot.permits(mcp_tool(server="after_sales"))


def test_permission_uses_original_identity_instead_of_public_alias(tmp_path):
    from app.tools.policy import PolicyStore

    path = tmp_path / "permissions.json"
    write_policy(path, {"public_alias": "read"})
    assert not PolicyStore(path).load().permits(mcp_tool())


def test_permission_changes_are_visible_without_recreating_store(tmp_path):
    from app.tools.policy import PolicyStore

    path = tmp_path / "permissions.json"
    store = PolicyStore(path)
    write_policy(path, {"query_logistics": "read"})
    old = store.load()
    write_policy(path, {})
    assert not store.load().permits(mcp_tool())
    assert old.permits(mcp_tool())
    write_policy(path, {"query_logistics": "read"})
    assert store.load().permits(mcp_tool())


@pytest.mark.parametrize("broken", ["{", "[]", "{}", '{"mcp": []}',
    '{"mcp": {"logistics": {"query_logistics": true}}}',
    '{"mcp": {"logistics": {"query_logistics": "admin"}}}',
    '{"mcp": {"logistics": {"query_logistics": "read"}, "bad": []}}',
])
def test_corrupt_configuration_revokes_previous_authorization(tmp_path, broken):
    from app.tools.policy import PolicyStore

    path = tmp_path / "permissions.json"
    store = PolicyStore(path)
    write_policy(path, {"query_logistics": "read"})
    assert store.load().permits(mcp_tool())
    path.write_text(broken, encoding="utf-8")
    snapshot = store.load()
    assert not snapshot.permits(mcp_tool())
    assert snapshot.permits(replace(mcp_tool(), source="builtin", server=None, original_name=None))


def test_missing_permission_file_revokes_previous_authorization(tmp_path):
    from app.tools.policy import PolicyStore

    path = tmp_path / "permissions.json"
    store = PolicyStore(path)
    write_policy(path, {"query_logistics": "read"})
    assert store.load().permits(mcp_tool())
    path.unlink()
    assert not store.load().permits(mcp_tool())


@pytest.mark.parametrize("permission,effect", [("write", "write"), ("read", "write"), ("write", "read")])
def test_external_writes_are_denied_even_if_locally_listed(tmp_path, permission, effect):
    from app.tools.policy import PolicyStore

    path = tmp_path / "permissions.json"
    write_policy(path, {"query_logistics": permission})
    assert not PolicyStore(path).load().permits(mcp_tool(effect=effect))


def test_shipped_policy_authorizes_the_three_read_tools():
    from app.tools.policy import PolicyStore

    snapshot = PolicyStore(Path("config/tool_permissions.json")).load()
    for server, name in [("logistics", "query_logistics"), ("after_sales", "query_warranty"),
                         ("after_sales", "query_return_progress")]:
        assert snapshot.permits(mcp_tool(server=server, original_name=name))
