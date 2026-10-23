"""ch08 统一结果格式化与审计摘要。

- format_outcome：把 ToolOutcome 序列化成中文可读 JSON（ensure_ascii=False），
  超长时丢弃大部分 data、保留白名单身份字段并加 truncated=true，仍保证合法 JSON。
- summarize_for_audit：把结果内容压缩为受限长度的审计摘要，不产生非法 JSON。
"""

import json

from app.tools.contracts import ToolOutcome

_IDENTITY_KEYS = ("ticket_no", "tracking_no", "return_no", "state", "status")


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def format_outcome(outcome: ToolOutcome, *, max_chars: int = 4000) -> str:
    """序列化 ToolOutcome 为中文 JSON 字符串。

    结果必定能被 json.loads 解析；超过 max_chars 时按字段裁剪 data，
    保留 ok/category/message 与白名单身份字段，绝不切断 JSON。
    """
    payload = {
        "ok": outcome.category in {"success", "not_found"},
        "category": outcome.category,
        "data": outcome.data,
        "message": outcome.message,
    }
    if outcome.actions:
        payload["actions"] = outcome.actions

    content = _dump(payload)
    if len(content) <= max_chars:
        return content

    data = outcome.data
    if isinstance(data, dict):
        payload["data"] = {k: data[k] for k in _IDENTITY_KEYS if k in data} or None
    else:
        payload["data"] = None
    payload["truncated"] = True
    return _dump(payload)


def summarize_for_audit(content: str, *, limit: int = 1000) -> str:
    """把结果内容压缩为审计摘要；不产生非法 JSON，也不携带超大 data。"""
    if len(content) <= limit:
        return content

    try:
        parsed = json.loads(content)
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        parsed = {}

    summary = {
        "ok": bool(parsed.get("ok")),
        "category": parsed.get("category", "fault"),
        "message": parsed.get("message"),
        "truncated": True,
    }
    result = _dump(summary)
    if len(result) > limit:
        # 极端情况：message 本身过大，压缩掉 message 兜底保持受限且合法。
        summary["message"] = None
        result = _dump(summary)
    return result
