"""LangChain 回调聚合：按 turn_token 汇总每轮 token、成本与阶段耗时。"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class TurnMetricsHandler:
    def __init__(self, *, prices, counter, trace_id_provider=None):
        self._prices = prices
        self._counter = counter
        self._trace_id_provider = trace_id_provider
        self._lock = threading.Lock()
        self._turns: dict[str, dict] = {}
        self._runs: dict[str, dict] = {}
        self._chains: dict[str, str | None] = {}
        self._run_tokens: dict[str, str] = {}

    def _token_from(self, metadata, parent_run_id):
        token = (metadata or {}).get("turn_token")
        if token:
            return str(token)
        if parent_run_id is None:
            return None
        parent = str(parent_run_id)
        return self._run_tokens.get(parent) or self._chains.get(parent)

    def _ensure(self, token: str) -> dict:
        return self._turns.setdefault(token, {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "usage_estimated": False,
            "model_usage": {},
            "cost_cny": 0.0,
            "first_token_ms": None,
            "unpriced_models": set(),
            "stages": {},
            "trace_id": None,
        })

    @staticmethod
    def _model_entry(acc: dict, model: str) -> dict:
        return acc["model_usage"].setdefault(model, {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_cny": 0.0,
            "estimated": False,
            "priced": True,
        })

    def _trace_id(self, metadata):
        for key in ("langfuse_trace_id", "trace_id"):
            value = (metadata or {}).get(key)
            if value:
                return str(value)
        if self._trace_id_provider is None:
            return None
        try:
            value = self._trace_id_provider()
            return str(value) if value else None
        except Exception:
            return None

    def _start_model_run(
        self, serialized, *, run_id, parent_run_id, metadata, prompt_tokens
    ):
        token = self._token_from(metadata, parent_run_id)
        if token is None:
            return
        model = (
            (serialized or {}).get("kwargs", {}).get("model")
            or (serialized or {}).get("name")
            or "unknown"
        )
        with self._lock:
            acc = self._ensure(token)
            self._runs[str(run_id)] = {
                "token": token,
                "model": str(model),
                "prompt_tokens": int(prompt_tokens),
                "started": time.perf_counter(),
                "first_token_ms": None,
            }
            self._run_tokens[str(run_id)] = token
            acc["llm_calls"] += 1
            trace_id = self._trace_id(metadata)
            if trace_id and not acc["trace_id"]:
                acc["trace_id"] = trace_id

    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs
    ):
        try:
            token = self._token_from(metadata, parent_run_id)
            if token is None:
                return
            with self._lock:
                self._chains[str(run_id)] = token
                acc = self._ensure(token)
                trace_id = self._trace_id(metadata)
                if trace_id and not acc["trace_id"]:
                    acc["trace_id"] = trace_id
        except Exception:
            logger.debug("chain 回调聚合失败", exc_info=True)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        try:
            with self._lock:
                self._chains.pop(str(run_id), None)
        except Exception:
            logger.debug("chain end 清理失败", exc_info=True)

    def on_chain_error(self, error, *, run_id, **kwargs):
        try:
            with self._lock:
                self._chains.pop(str(run_id), None)
        except Exception:
            logger.debug("chain error 清理失败", exc_info=True)

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None,
        metadata=None, **kwargs,
    ):
        try:
            prompt_tokens = sum(
                self._counter.count_messages(batch) for batch in (messages or [])
            )
            self._start_model_run(
                serialized, run_id=run_id, parent_run_id=parent_run_id,
                metadata=metadata, prompt_tokens=prompt_tokens,
            )
        except Exception:
            logger.debug("chat model start 聚合失败", exc_info=True)

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None,
        metadata=None, **kwargs,
    ):
        try:
            prompt_tokens = sum(
                self._counter.count_text(prompt or "") for prompt in (prompts or [])
            )
            self._start_model_run(
                serialized, run_id=run_id, parent_run_id=parent_run_id,
                metadata=metadata, prompt_tokens=prompt_tokens,
            )
        except Exception:
            logger.debug("llm start 聚合失败", exc_info=True)

    def on_llm_new_token(self, token, *, run_id, **kwargs):
        try:
            with self._lock:
                info = self._runs.get(str(run_id))
                if info is not None and info["first_token_ms"] is None:
                    info["first_token_ms"] = int(
                        (time.perf_counter() - info["started"]) * 1000
                    )
        except Exception:
            logger.debug("首 token 聚合失败", exc_info=True)

    def on_llm_end(self, response, *, run_id, **kwargs):
        try:
            with self._lock:
                info = self._runs.pop(str(run_id), None)
                if info is None:
                    return
                self._run_tokens.pop(str(run_id), None)
                acc = self._ensure(info["token"])
                message = None
                generation = None
                try:
                    generation = response.generations[0][0]
                    message = getattr(generation, "message", None)
                except Exception:
                    pass
                usage = getattr(message, "usage_metadata", None) or {}
                prompt = usage.get("input_tokens")
                completion = usage.get("output_tokens")
                total = usage.get("total_tokens")
                try:
                    token_usage = (
                        getattr(response, "llm_output", None) or {}
                    ).get("token_usage") or {}
                except Exception:
                    token_usage = {}
                if prompt is None:
                    prompt = token_usage.get("prompt_tokens")
                if completion is None:
                    completion = token_usage.get("completion_tokens")
                if total is None:
                    total = token_usage.get("total_tokens")
                estimated = prompt is None or completion is None
                if prompt is None:
                    prompt = info["prompt_tokens"]
                if completion is None:
                    text = (
                        getattr(message, "content", "")
                        if message is not None
                        else getattr(generation, "text", "")
                    )
                    completion = self._counter.count_text(str(text or ""))
                prompt = int(prompt or 0)
                completion = int(completion or 0)
                total = int(total) if total is not None else prompt + completion
                acc["prompt_tokens"] += prompt
                acc["completion_tokens"] += completion
                acc["total_tokens"] += total
                acc["usage_estimated"] = acc["usage_estimated"] or estimated
                model = info["model"]
                entry = self._model_entry(acc, model)
                entry["calls"] += 1
                entry["prompt_tokens"] += prompt
                entry["completion_tokens"] += completion
                entry["estimated"] = entry["estimated"] or estimated
                price = self._prices.get(model)
                cost = self._prices.cost(model, prompt, completion)
                acc["cost_cny"] = round(acc["cost_cny"] + cost, 6)
                entry["cost_cny"] = round(entry["cost_cny"] + cost, 6)
                entry["priced"] = price is not None
                if price is None:
                    acc["unpriced_models"].add(model)
                if info["first_token_ms"] is not None and acc["first_token_ms"] is None:
                    acc["first_token_ms"] = info["first_token_ms"]
        except Exception:
            logger.debug("llm end 聚合失败", exc_info=True)

    def on_llm_error(self, error, *, run_id, **kwargs):
        try:
            with self._lock:
                self._runs.pop(str(run_id), None)
                self._run_tokens.pop(str(run_id), None)
        except Exception:
            logger.debug("llm error 清理失败", exc_info=True)

    def record_stage(
        self, turn_token, kind, *, model, duration_ms, texts=None, items=0
    ):
        try:
            tokens = sum(
                self._counter.count_text(text or "") for text in (texts or [])
            )
            with self._lock:
                acc = self._ensure(str(turn_token))
                stage = acc["stages"].setdefault(kind, {
                    "calls": 0,
                    "duration_ms": 0,
                    "items": 0,
                    "tokens": 0,
                    "estimated": True,
                    "model": None,
                })
                stage["calls"] += 1
                stage["duration_ms"] += int(duration_ms)
                stage["items"] += int(items)
                stage["tokens"] += tokens
                if stage["model"] is None:
                    stage["model"] = str(model)
                entry = self._model_entry(acc, str(model))
                entry["calls"] += 1
                entry["prompt_tokens"] += tokens
                entry["estimated"] = True
                price = self._prices.get(str(model))
                cost = self._prices.cost(str(model), tokens, 0)
                acc["cost_cny"] = round(acc["cost_cny"] + cost, 6)
                entry["cost_cny"] = round(entry["cost_cny"] + cost, 6)
                entry["priced"] = price is not None
                if price is None:
                    acc["unpriced_models"].add(str(model))
                acc["usage_estimated"] = True
        except Exception:
            logger.debug("阶段指标聚合失败", exc_info=True)

    def finish_turn(self, turn_token):
        with self._lock:
            acc = self._turns.pop(str(turn_token), None)
            if acc is None:
                return None
            return {
                "llm_calls": acc["llm_calls"],
                "prompt_tokens": acc["prompt_tokens"],
                "completion_tokens": acc["completion_tokens"],
                "total_tokens": acc["total_tokens"],
                "usage_estimated": acc["usage_estimated"],
                "model_usage": acc["model_usage"],
                "cost_cny": acc["cost_cny"],
                "first_token_ms": acc["first_token_ms"],
                "unpriced_models": sorted(acc["unpriced_models"]),
                "stages": acc["stages"],
                "trace_id": acc["trace_id"],
            }
