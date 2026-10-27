"""意图成本账：按 `intent:<x>` 标签汇总 token 与成本。

数据源是 Langfuse v2 Metrics API（observations 视图）。意图标签由
`IntentTaggingCallbackHandler` 打在「意图确定之后」创建的观测上，因此这里
的口径与观测面板一致：只算分类之后的花销；指代消解、意图分类本身落在
「意图未定」一栏，单独列出来，不混进任何意图。

请求数取根观测（`handle-chat-turn`）的计数：一轮对话一个根 span，标签与
它的生成观测同组，按 tags 分组后自然对齐。

产物落盘 `data/cost_report/latest.json`：页面只读产物，重跑由页面按钮或
`make cost-report` 触发。
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

REPORT_PATH = Path("data/cost_report/latest.json")
DEFAULT_WINDOW_DAYS = 7
ROOT_SPAN_NAME = "handle-chat-turn"
INTENT_PREFIX = "intent:"

INTENT_LABELS = {
    "logistics": "物流",
    "order": "订单",
    "product_consult": "商品咨询",
    "refund_return": "退款退货",
    "after_sale": "售后",
    "complaint": "投诉",
    "human": "转人工",
    "chitchat": "闲聊",
    "other": "其他",
}

QueryFn = Callable[[dict], list[dict]]


class CostReportError(RuntimeError):
    """取不到 Langfuse 数据；消息直接展示给页面，不含任何凭据。"""


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _environment() -> str:
    return (settings.app_env or "").strip() or "development"


def _query_metrics(query: dict) -> list[dict]:
    base_url = settings.langfuse_base_url.rstrip("/")
    try:
        resp = httpx.get(
            f"{base_url}/api/public/v2/metrics",
            params={"query": json.dumps(query, ensure_ascii=False)},
            auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise CostReportError(f"连不上 Langfuse：{type(exc).__name__}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("message") or ""
        except ValueError:
            detail = ""
        suffix = f"：{detail}" if detail else ""
        raise CostReportError(f"Langfuse 指标接口返回 HTTP {resp.status_code}{suffix}")
    try:
        return list(resp.json().get("data") or [])
    except ValueError as exc:
        raise CostReportError("Langfuse 指标接口返回的不是 JSON") from exc


def _intent_of(tags: list[str]) -> str | None:
    for tag in tags:
        if tag.startswith(INTENT_PREFIX):
            intent = tag[len(INTENT_PREFIX):]
            if intent:
                return intent
    return None


def _tags_of(row: dict) -> list[str]:
    return [str(tag) for tag in row.get("tags") or ()]


def _number(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _rank(entries: list[dict]) -> dict | None:
    if not entries:
        return None
    priced = any(entry["cost"] > 0 for entry in entries)
    key = "cost" if priced else "tokens"
    top = max(entries, key=lambda entry: (entry[key], entry["tokens"]))
    return {
        "intent": top["intent"],
        "label": top["label"],
        "requests": top["requests"],
        "tokens": top["tokens"],
        "cost": top["cost"],
        "ranked_by": key,
    }


def aggregate(
    tag_rows: list[dict],
    root_rows: list[dict],
    *,
    window_days: int,
    generated_at: str,
    environment: str,
) -> dict:
    """把 Metrics API 的两组行合成成本账。纯函数，便于单测。"""
    tokens: dict[str | None, int] = defaultdict(int)
    cost: dict[str | None, float] = defaultdict(float)
    for row in tag_rows:
        intent = _intent_of(_tags_of(row))
        tokens[intent] += int(_number(row, "sum_totalTokens"))
        cost[intent] += _number(row, "sum_totalCost")

    requests: dict[str | None, int] = defaultdict(int)
    for row in root_rows:
        requests[_intent_of(_tags_of(row))] += int(_number(row, "count_count"))

    names = sorted({key for key in (*tokens, *requests) if key})
    total_tokens = sum(tokens[name] for name in names)
    entries = []
    for name in names:
        entry_tokens = tokens[name]
        entry_requests = requests[name]
        entry_cost = cost[name]
        entries.append({
            "intent": name,
            "label": INTENT_LABELS.get(name, name),
            "requests": entry_requests,
            "tokens": entry_tokens,
            "cost": round(entry_cost, 9),
            "avg_tokens": round(entry_tokens / entry_requests) if entry_requests else 0,
            "share": round(entry_tokens / total_tokens, 4) if total_tokens else 0.0,
            # token 花出去了却没有成本，说明模型没匹配到价格，页面要提示
            "priced": not (entry_tokens > 0 and entry_cost == 0),
        })
    entries.sort(key=lambda entry: (-entry["tokens"], -entry["cost"], entry["intent"]))

    total_requests = sum(entry["requests"] for entry in entries)
    total_cost = sum(entry["cost"] for entry in entries)
    warnings: list[str] = []
    unpriced = [entry["label"] for entry in entries if not entry["priced"]]
    if unpriced:
        warnings.append(
            "这些意图的 token 没有匹配到模型价格，成本按 0 显示：" + "、".join(unpriced)
        )
    if not entries:
        warnings.append("窗口内没有带意图标签的请求，先聊几轮再看")

    return {
        "generated_at": generated_at,
        "window_days": window_days,
        "environment": environment,
        "source": "langfuse",
        "root_span": ROOT_SPAN_NAME,
        "totals": {
            "requests": total_requests,
            "tokens": total_tokens,
            "cost": round(total_cost, 9),
            "avg_tokens": round(total_tokens / total_requests) if total_requests else 0,
        },
        "top_intent": _rank(entries),
        "intents": entries,
        "untagged": {
            "requests": requests.get(None, 0),
            "tokens": tokens.get(None, 0),
            "cost": round(cost.get(None, 0.0), 9),
        },
        "warnings": warnings,
    }


def build_report(
    days: int = DEFAULT_WINDOW_DAYS,
    *,
    now: datetime | None = None,
    query: QueryFn | None = None,
    environment: str | None = None,
) -> dict:
    if days < 1:
        raise ValueError("days 至少为 1")
    query = query or _query_metrics
    environment = environment or _environment()
    now = now or datetime.now(timezone.utc)
    from_ts = _iso(now - timedelta(days=days))
    to_ts = _iso(now)

    env_filter = {
        "type": "string", "column": "environment", "operator": "=", "value": environment,
    }
    base = {
        "view": "observations",
        "fromTimestamp": from_ts,
        "toTimestamp": to_ts,
    }
    tag_rows = query({
        **base,
        "dimensions": [{"field": "tags"}],
        "metrics": [
            {"measure": "totalTokens", "aggregation": "sum"},
            {"measure": "totalCost", "aggregation": "sum"},
        ],
        "filters": [env_filter],
    })
    root_rows = query({
        **base,
        "dimensions": [{"field": "tags"}],
        "metrics": [{"measure": "count", "aggregation": "count"}],
        "filters": [
            env_filter,
            {
                "type": "string", "column": "name", "operator": "=",
                "value": ROOT_SPAN_NAME,
            },
        ],
    })
    return aggregate(
        tag_rows,
        root_rows,
        window_days=days,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        environment=environment,
    )


def write_report(report: dict, path: Path | None = None) -> Path:
    path = path or REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_report(path: Path | None = None) -> dict | None:
    path = path or REPORT_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("成本报表产物读不出来，按无产物处理：%s", path, exc_info=True)
        return None


def refresh(
    days: int = DEFAULT_WINDOW_DAYS,
    *,
    query: QueryFn | None = None,
    now: datetime | None = None,
    path: Path | None = None,
) -> dict:
    """重跑并落盘。页面按钮和 CLI 都走这里，保证口径只有一套。"""
    report = build_report(days, now=now, query=query)
    write_report(report, path)
    return report
