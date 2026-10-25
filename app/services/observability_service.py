"""ch09 观测聚合服务：把 turn_metrics / 工具审计 / 反馈聚合成 dashboard 数据。

只读本地库，不依赖 Langfuse；trace 深链仅当配置了 langfuse_project_id 才产出。
"""

import logging
from datetime import datetime, timedelta
from math import ceil

from sqlalchemy import func, select

from app.config import settings as default_settings
from app.db.models import (
    AnswerFeedback,
    AuditStatus,
    ToolAuditLog,
    TurnMetrics,
    TurnStatus,
)

logger = logging.getLogger(__name__)


def percentile(values, p):
    """nearest-rank 百分位；空序列返回 None。"""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, ceil(p / 100 * len(ordered)) - 1)
    return ordered[index]


class ObservabilityService:
    def __init__(self, session_factory, settings=None):
        self._session_factory = session_factory
        self._settings = settings or default_settings

    @staticmethod
    def _status_value(status) -> str:
        return status.value if hasattr(status, "value") else str(status)

    @staticmethod
    def _rating_value(rating) -> str:
        return rating.value if hasattr(rating, "value") else str(rating)

    def _since(self, days: int) -> datetime:
        return datetime.now() - timedelta(days=days)

    def _trace_url(self, trace_id):
        if not trace_id:
            return None
        project = (self._settings.langfuse_project_id or "").strip()
        if not project:
            return None
        base = (self._settings.langfuse_base_url or "").rstrip("/")
        return f"{base}/project/{project}/traces/{trace_id}"

    @staticmethod
    def _merge_models(models, model_usage, unpriced):
        for name, entry in (model_usage or {}).items():
            if not isinstance(entry, dict):
                continue
            acc = models.setdefault(str(name), {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "cost_cny": 0.0, "priced": True, "estimated": False,
            })
            acc["calls"] += int(entry.get("calls") or 0)
            acc["prompt_tokens"] += int(entry.get("prompt_tokens") or 0)
            acc["completion_tokens"] += int(entry.get("completion_tokens") or 0)
            acc["cost_cny"] = round(
                acc["cost_cny"] + float(entry.get("cost_cny") or 0), 6
            )
            # priced 是所有条目 priced 的 AND；estimated 是 OR
            acc["priced"] = acc["priced"] and bool(entry.get("priced", True))
            acc["estimated"] = acc["estimated"] or bool(entry.get("estimated"))
            if not bool(entry.get("priced", True)):
                unpriced.add(str(name))

    @staticmethod
    def _merge_stages(stages, row_stages):
        for kind, stage in (row_stages or {}).items():
            if not isinstance(stage, dict):
                continue
            acc = stages.setdefault(str(kind), {
                "calls": 0, "duration_ms": 0, "items": 0, "tokens": 0,
                "estimated": False, "model": None,
            })
            acc["calls"] += int(stage.get("calls") or 0)
            acc["duration_ms"] += int(stage.get("duration_ms") or 0)
            acc["items"] += int(stage.get("items") or 0)
            acc["tokens"] += int(stage.get("tokens") or 0)
            acc["estimated"] = acc["estimated"] or bool(stage.get("estimated"))
            if acc["model"] is None:
                acc["model"] = stage.get("model")

    @staticmethod
    def _stage_entries(stages):
        out = {}
        for kind, acc in stages.items():
            calls = acc["calls"]
            out[kind] = {
                "calls": calls,
                "duration_ms": acc["duration_ms"],
                "items": acc["items"],
                "tokens": acc["tokens"],
                "estimated": acc["estimated"],
                "model": acc["model"],
                "avg_duration_ms": round(acc["duration_ms"] / calls, 1) if calls else None,
                "avg_items": round(acc["items"] / calls, 1) if calls else None,
                "avg_tokens": round(acc["tokens"] / calls, 1) if calls else None,
            }
        return out

    async def summary(self, days: int) -> dict:
        since = self._since(days)
        async with self._session_factory() as session:
            rows = (await session.execute(
                select(TurnMetrics)
                .where(TurnMetrics.created_at >= since)
                .order_by(TurnMetrics.created_at.desc())
                .limit(5000)
            )).scalars().all()
            feedback_rows = (await session.execute(
                select(AnswerFeedback).where(AnswerFeedback.created_at >= since)
            )).scalars().all()

        turns = len(rows)
        up = sum(1 for f in feedback_rows if self._rating_value(f.rating) == "up")
        down = sum(1 for f in feedback_rows if self._rating_value(f.rating) == "down")

        status_counts: dict[str, int] = {}
        refusal_sources: dict[str, int] = {}
        models: dict[str, dict] = {}
        stages: dict[str, dict] = {}
        unpriced: set[str] = set()
        estimated_turns = 0
        failures = 0  # §10.2：错误率 = (error + cancelled) / turns；interrupted 不计入
        refusals = 0
        prompt_total = completion_total = total_total = 0
        cost_total = 0.0
        latencies: list[int] = []
        first_tokens: list[int] = []

        for row in rows:
            status = self._status_value(row.status)
            status_counts[status] = status_counts.get(status, 0) + 1
            if status in ("error", "cancelled"):
                failures += 1
            if row.refusal_source:
                refusal_sources[row.refusal_source] = (
                    refusal_sources.get(row.refusal_source, 0) + 1
                )
                refusals += 1
            prompt_total += int(row.prompt_tokens or 0)
            completion_total += int(row.completion_tokens or 0)
            total_total += int(row.total_tokens or 0)
            cost_total += float(row.cost_cny or 0)
            if row.usage_estimated:
                estimated_turns += 1
            if row.latency_ms is not None:
                latencies.append(int(row.latency_ms))
            if row.first_token_ms is not None:
                first_tokens.append(int(row.first_token_ms))
            self._merge_models(models, row.model_usage, unpriced)
            self._merge_stages(stages, row.stages)

        return {
            "window_days": days,
            "turns": turns,
            "satisfaction": {
                "up": up,
                "down": down,
                "rate": round(up / (up + down), 4) if (up + down) else None,
            },
            "refusal_rate": round(refusals / turns, 4) if turns else None,
            "error_rate": round(failures / turns, 4) if turns else None,
            "status_counts": status_counts,
            "refusal_sources": refusal_sources,
            "tokens": {
                "prompt": prompt_total,
                "completion": completion_total,
                "total": total_total,
            },
            "cost_cny": round(cost_total, 6),
            "estimated_turns": estimated_turns,
            "unpriced_models": sorted(unpriced),
            "models": [
                {"model": name, **entry}
                for name, entry in sorted(models.items())
            ],
            "stages": self._stage_entries(stages),
            "latency_ms": {
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "avg": round(sum(latencies) / len(latencies), 1) if latencies else None,
            },
            "first_token_ms": {
                "p50": percentile(first_tokens, 50),
                "p95": percentile(first_tokens, 95),
                "avg": round(sum(first_tokens) / len(first_tokens), 1) if first_tokens else None,
            },
        }

    def _serialize_turn(self, row) -> dict:
        return {
            "id": row.id,
            "conversation_id": row.conversation_id,
            "turn_token": row.turn_token,
            "trace_id": row.trace_id,
            "trace_url": self._trace_url(row.trace_id),
            "route": row.route,
            "intent": row.intent,
            "status": self._status_value(row.status),
            "error_type": row.error_type,
            "llm_calls": row.llm_calls,
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": row.completion_tokens,
            "total_tokens": row.total_tokens,
            "usage_estimated": row.usage_estimated,
            "cost_cny": float(row.cost_cny or 0),
            "latency_ms": row.latency_ms,
            "first_token_ms": row.first_token_ms,
            "refusal_source": row.refusal_source,
            "citations_count": row.citations_count,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    async def turns(self, days, limit, offset, status=None, route=None) -> dict:
        since = self._since(days)
        filters = [TurnMetrics.created_at >= since]
        if status:
            filters.append(TurnMetrics.status == TurnStatus(status))
        if route:
            filters.append(TurnMetrics.route == route)
        async with self._session_factory() as session:
            total = (
                await session.execute(
                    select(func.count()).select_from(TurnMetrics).where(*filters)
                )
            ).scalar() or 0
            rows = (await session.execute(
                select(TurnMetrics)
                .where(*filters)
                .order_by(TurnMetrics.created_at.desc(), TurnMetrics.id.desc())
                .limit(limit)
                .offset(offset)
            )).scalars().all()
        return {
            "items": [self._serialize_turn(r) for r in rows],
            "total": int(total),
            "days": days,
            "limit": limit,
            "offset": offset,
        }

    async def tools(self, days) -> dict:
        since = self._since(days)
        async with self._session_factory() as session:
            summary_rows = (await session.execute(
                select(
                    ToolAuditLog.tool_name,
                    func.count().label("calls"),
                    func.avg(ToolAuditLog.duration_ms).label("avg_duration_ms"),
                    func.sum(ToolAuditLog.retry_count).label("retry_total"),
                )
                .where(ToolAuditLog.created_at >= since)
                .group_by(ToolAuditLog.tool_name)
            )).all()
            failure_rows = (await session.execute(
                select(ToolAuditLog.tool_name, func.count())
                .where(
                    ToolAuditLog.created_at >= since,
                    ToolAuditLog.status != AuditStatus.SUCCESS,
                )
                .group_by(ToolAuditLog.tool_name)
            )).all()
        failures = {str(name): int(count) for name, count in failure_rows}
        items = []
        for name, calls, avg_ms, retry_total in summary_rows:
            items.append({
                "tool_name": str(name),
                "calls": int(calls or 0),
                "failures": failures.get(str(name), 0),
                "avg_duration_ms": round(float(avg_ms), 1) if avg_ms is not None else None,
                "retry_total": int(retry_total or 0),
            })
        return {"items": items, "total": len(items), "days": days}
