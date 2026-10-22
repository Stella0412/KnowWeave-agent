"""中文感知的 token 估算与从模型窗口倒推的上下文预算。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass

import tiktoken
from langchain_core.messages import BaseMessage, ToolMessage

_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")


class TokenCounter:
    """CJK 按字符计，非 CJK 走 cl100k_base，最后统一乘估算系数。"""

    def __init__(self, *, cjk_tokens_per_char: float = 1.0, estimate_factor: float = 1.15):
        self.cjk_tokens_per_char = float(cjk_tokens_per_char)
        self.estimate_factor = float(estimate_factor)
        self._enc = tiktoken.get_encoding("cl100k_base")

    def count_text(self, text: str) -> int:
        text = text or ""
        if not text:
            return 0
        cjk = len(_CJK_RE.findall(text))
        rest = _CJK_RE.sub("", text)
        raw = cjk * self.cjk_tokens_per_char + len(self._enc.encode(rest))
        return max(1, math.ceil(raw * self.estimate_factor))

    def count_message(self, message: BaseMessage) -> int:
        total = 4  # 每条消息协议开销
        total += self.count_text(text_of(getattr(message, "content", "")))
        name = getattr(message, "name", None)
        if name:
            total += self.count_text(str(name))
        for call in getattr(message, "tool_calls", None) or []:
            total += self.count_text(json.dumps(call, ensure_ascii=False, sort_keys=True, default=str))
        if isinstance(message, ToolMessage):
            total += self.count_text(message.tool_call_id or "")
        return total

    def count_messages(self, messages: list[BaseMessage]) -> int:
        return sum(self.count_message(message) for message in messages or [])

    def __call__(self, messages: list[BaseMessage]) -> int:
        return self.count_messages(messages)

    def truncate_text(self, text: str, max_tokens: int, marker: str = "…（已截断）") -> str:
        if max_tokens <= 0:
            return ""
        if self.count_text(text) <= max_tokens:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.count_text(text[:mid] + marker) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + marker


@dataclass(frozen=True)
class ContextBudget:
    model_window: int
    fixed: int
    peak: int
    window_history: int
    history: int
    layer1: int
    layer2: int

    def as_dict(self) -> dict:
        return asdict(self)


class ContextBudgetError(RuntimeError):
    pass


def compute_budget(settings) -> ContextBudget:
    share = float(settings.layer1_share)
    if not 0 < share < 1:
        raise ContextBudgetError("LAYER1_SHARE 必须在 0 和 1 之间")
    fixed = (
        settings.system_prefix_reserve_tokens
        + settings.rerank_top_n * settings.evidence_tokens_per_chunk
        + settings.summary_inject_max_tokens
        + settings.max_output_tokens
        + settings.max_user_input_tokens
        + settings.context_safety_margin_tokens
    )
    peak = settings.max_agent_steps * settings.tool_result_max_tokens
    window_history = settings.model_context_window - fixed - peak
    rounds_history = math.ceil(
        settings.context_target_rounds * settings.steady_turn_tokens / share
    )
    history = min(window_history, rounds_history)
    layer1 = int(history * share)
    layer2 = int(history * (1 - share))
    if window_history <= 0 or history <= 0 or layer1 < settings.steady_turn_tokens:
        raise ContextBudgetError(
            f"上下文预算不足：window_history={window_history} history={history} "
            f"layer1={layer1} steady={settings.steady_turn_tokens}"
        )
    return ContextBudget(
        model_window=settings.model_context_window,
        fixed=fixed,
        peak=peak,
        window_history=window_history,
        history=history,
        layer1=layer1,
        layer2=layer2,
    )
