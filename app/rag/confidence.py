"""证据置信度：四信号加权，兜底闸门按这条线划。

权重是**代码里的常量**（不随评估集调），校准校的是 `confidence_min_score`
这条线划在哪里 —— 见 `eval/confidence_calibration.py` 与评估页的校准面板。

| 信号 | 权重 | 含义 |
|---|---|---|
| Top1 精排分 | 0.5 | 最相关那条证据的交叉编码器分（0~1） |
| 有效证据数 | 0.2 | 精排分 ≥ `EFFECTIVE_SCORE_FLOOR` 的条数，`EFFECTIVE_COUNT_CAP` 条封顶 |
| Top1-Top2 分差 | 0.2 | 绝对分差 `MARGIN_CAP` 封顶 —— 只差一点说明没有明显更优的证据 |
| 关键条款命中 | 0.1 | 问题里的关键词组有多少出现在前 3 条证据的**问法**里 |

关键条款用字面重合（中文二字组 + 英文/型号词）近似：用户措辞与库里问法离得远时
（库里写「猫窝清洗保养说明」，用户问「能扔洗衣机里洗吗」），这一项会掉到 0 ——
这正是「可答被误拦」的成因，也是它只占 0.1 的原因，别让字面差异一票否决。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.rag.evidence import Evidence

WEIGHTS = {
    "top1": 0.5,
    "effective": 0.2,
    "margin": 0.2,
    "clause": 0.1,
}

# 精排分低于这条线的不算「有效证据」：一个 0.02 分的 chunk 顶不上一条真证据。
EFFECTIVE_SCORE_FLOOR = 0.1
EFFECTIVE_COUNT_CAP = 3
# Top1 与 Top2 差到 0.2 就算把其余候选甩开；再大也不再额外加分。
MARGIN_CAP = 0.2
CLAUSE_TOP_N = 3

_STOP_CHARS = set(
    "的了吗呢吧啊呀哦嗯是在有和与及或就都也还把被给对从到为以于而但如果不没很太最更"
    "可以能会要得着过个这那我你他她它们怎样什么如何请问一下些之其"
)
_ASCII_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[一-鿿]+")


@dataclass(frozen=True)
class ConfidenceSignals:
    top1: float
    effective: float
    margin: float
    clause: float
    score: float
    evidence_count: int

    def as_dict(self) -> dict:
        return {
            "top1": round(self.top1, 4),
            "effective": round(self.effective, 4),
            "margin": round(self.margin, 4),
            "clause": round(self.clause, 4),
            "score": round(self.score, 4),
            "evidence_count": self.evidence_count,
        }


def terms(text: str) -> set[str]:
    """字面关键词组：英文/型号词整词，中文取二字组，丢掉虚词单字。"""
    lowered = (text or "").lower()
    out = {word for word in _ASCII_WORD.findall(lowered) if len(word) >= 2}
    for run in _CJK_RUN.findall(lowered):
        if len(run) == 1:
            if run not in _STOP_CHARS:
                out.add(run)
            continue
        for index in range(len(run) - 1):
            pair = run[index:index + 2]
            if pair[0] in _STOP_CHARS and pair[1] in _STOP_CHARS:
                continue
            out.add(pair)
    return {term for term in out if term not in _STOP_CHARS}


def clause_hit(query: str, items: list[Evidence], top_n: int = CLAUSE_TOP_N) -> float:
    """问题关键词组在前 `top_n` 条证据**问法**里的覆盖比例。"""
    wanted = terms(query)
    if not wanted:
        return 0.0
    haystack: set[str] = set()
    for item in items[:top_n]:
        haystack |= terms(item.question)
    return len(wanted & haystack) / len(wanted)


def effective_count(scores: list[float]) -> float:
    above = sum(1 for score in scores if score >= EFFECTIVE_SCORE_FLOOR)
    return min(above, EFFECTIVE_COUNT_CAP) / EFFECTIVE_COUNT_CAP


def margin_score(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    gap = max(scores[0] - scores[1], 0.0)
    return min(gap, MARGIN_CAP) / MARGIN_CAP


def signals(items: list[Evidence], query: str) -> ConfidenceSignals:
    """按精排分降序算四信号；`items` 应是一轮检索后真正进入 prompt 的候选。"""
    ordered = sorted(items, key=lambda item: item.score, reverse=True)
    scores = [float(item.score) for item in ordered]
    top1 = max(scores[0], 0.0) if scores else 0.0
    effective = effective_count(scores)
    margin = margin_score(scores)
    clause = clause_hit(query, ordered)
    score = (
        WEIGHTS["top1"] * top1
        + WEIGHTS["effective"] * effective
        + WEIGHTS["margin"] * margin
        + WEIGHTS["clause"] * clause
    )
    return ConfidenceSignals(
        top1=top1, effective=effective, margin=margin, clause=clause,
        score=min(max(score, 0.0), 1.0), evidence_count=len(scores),
    )


def threshold() -> float:
    return settings.confidence_min_score
