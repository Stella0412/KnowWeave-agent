"""导出前脱敏：手机号 / 身份证 / 邮箱 / 银行卡号。

Langfuse 的 `mask_otel_spans` 在导出阶段统一改写 OpenTelemetry span 属性，能覆盖
LangChain 回调自动采集的 prompt 与工具参数；`mask` 旧钩子只管 SDK 自己写入的数据。
"""

from __future__ import annotations

import re
from typing import Any, Optional

# 顺序有意义：先匹配 18 位身份证，避免被 16–19 位银行卡规则抢先截断。
_PATTERNS = (
    (re.compile(r"(?<![\d.])\d{17}[\dXx](?![\d.])"), "[REDACTED ID]"),
    (re.compile(r"(?<![\d.])(?:\+?86[- ]?)?1[3-9]\d{9}(?![\d.])"), "[REDACTED PHONE]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[REDACTED EMAIL]"),
    # 小数点两侧不参与匹配，否则 `0.9845678901234567` 这类分数会被当成卡号。
    (re.compile(r"(?<![\d.])(?:\d[ -]?){15,18}\d(?![\d.])"), "[REDACTED CARD]"),
)


def mask_text(value: str) -> str:
    for pattern, replacement in _PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def mask_otel_spans(*, params: Any) -> Optional[Any]:
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        replacements = {}
        for key, value in span.attributes.items():
            if isinstance(value, str):
                masked = mask_text(value)
                if masked != value:
                    replacements[key] = masked
        if replacements:
            patches[identifier] = OtelSpanPatch(set_attributes=replacements)
    return MaskOtelSpansResult(span_patches=patches) if patches else None
