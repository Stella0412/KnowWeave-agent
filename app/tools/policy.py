"""本地工具权限存储：热加载、逐工具授权。

- 权限以 MCP Server + 原始工具名为键，永不按公开别名匹配。
- 每次 load 全量重读并重新解析，出错即发布禁止 MCP 的快照（不沿用旧授权）。
- 内置工具始终放行，不受权限文件损坏影响。
- 本章仅外部读工具可放行：definition.effect 与配置权限都必须是 "read"。
- 未知工具默认拒绝。
"""

import json
from dataclasses import dataclass
from pathlib import Path

from app.tools.contracts import ToolDefinition

_VALID_PERMISSIONS = {"read", "write"}


@dataclass
class PolicySnapshot:
    """一份不可变授权快照。_permissions: {server: {original_name: permission}}。"""

    _permissions: dict

    def permits(self, definition: ToolDefinition) -> bool:
        if definition.source == "builtin":
            return True
        if definition.source != "mcp":
            return False
        server_perms = self._permissions.get(definition.server)
        if not isinstance(server_perms, dict):
            return False
        permission = server_perms.get(definition.original_name)
        return permission == "read" and definition.effect == "read"


class PolicyStore:
    """按路径读取本地权限文件；load() 每次都重新解析。"""

    def __init__(self, path) -> None:
        self.path = Path(path)

    def load(self) -> PolicySnapshot:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return PolicySnapshot(_parse(data))
        except Exception:
            return PolicySnapshot({})


def _parse(data) -> dict:
    """解析策略根对象，任一结构非法都抛 ValueError（触发整体拒绝 MCP）。"""
    if not isinstance(data, dict):
        raise ValueError("root must be an object")
    mcp = data.get("mcp")
    if not isinstance(mcp, dict):
        raise ValueError("mcp must be an object")

    permissions: dict = {}
    for server, tools in mcp.items():
        if not isinstance(server, str) or not isinstance(tools, dict):
            raise ValueError("server entry must map a string name to an object")
        per_tool: dict = {}
        for name, permission in tools.items():
            if not isinstance(name, str) or not isinstance(permission, str):
                raise ValueError("permission keys and values must be strings")
            if permission not in _VALID_PERMISSIONS:
                raise ValueError(f"unknown permission value: {permission!r}")
            per_tool[name] = permission
        permissions[server] = per_tool
    return permissions
