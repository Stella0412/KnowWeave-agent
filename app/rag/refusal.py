"""两段拒答闸门（spec §9.2）。

第一段在检索侧：早拒，省掉整轮生成与自评的开销。
第二段在生成前：模型自评「手上的资料够不够答」，不够就拒。

两条闸门分别落 `low_confidence_questions` 的两个 source，与 DDL 的枚举对齐。
**本模块只做判断，不碰数据库** —— 落池是 async 的，放在调用方（app/tools/business.py），
这样这段纯逻辑能单测，落池失败也不至于把判断逻辑搞成 async（Review Focus 6）。
"""

from dataclasses import dataclass

from app.config import settings
from app.rag.evidence import Evidence
from app.rag.strategies import VECTOR, has_rerank

SOURCE_RETRIEVAL = "retrieval_low_conf"
SOURCE_SELF_CHECK = "self_check"

# 拒答话术：明确说明没依据 + 给可转人工的出口。不硬编政策细节，也不许模型自己补常识
REFUSAL_TEXT = (
    "抱歉，知识库里没有查到相关依据，我无法确认这个问题的答案。"
    "建议您联系人工客服核实，人工客服服务时间为每日 9:00 至 22:00。"
)


@dataclass(frozen=True)
class RefusalDecision:
    refused: bool
    source: str | None = None
    reason: str | None = None


def retrieval_gate(
    strategy: str,
    candidates: list[Evidence],
    *,
    rerank_score: float | None = None,
    rerank_min_score: float | None = None,
    dense_min_score: float | None = None,
) -> RefusalDecision:
    """检索侧早拒。`rerank_score` 的语义**收紧为「精排真的跑成了」时才有值**：
 None 既代表「本策略不做精排」，也代表「精排上游降级」——两种情况都按纯 hybrid 规则办
（除空结果外不拒），绝不拿 RRF 名次分去比精排阈值（Ruling R1）。"""
    if not candidates:
        return RefusalDecision(True, SOURCE_RETRIEVAL, "召回结果为空：知识库里没有可用的候选")

    if has_rerank(strategy):
        threshold = settings.rerank_min_score if rerank_min_score is None else rerank_min_score
        # Ruling R1：rerank_score is None 表示「精排上游降级」，此时 candidates 的 score 是
        # hybrid_search 的 RRF 名次倒数（≈0.016~0.033），与精排分不同量纲 —— 拿它比 threshold
        # 会把每一道题都拒掉。降级等价于纯 hybrid：除空结果外不拒（与下面 BM25/HYBRID 同规则）。
        if rerank_score is None:
            return RefusalDecision(False)
        if rerank_score < threshold:
            return RefusalDecision(
                True, SOURCE_RETRIEVAL,
                f"精排最高分 {rerank_score:.4f} 低于阈值 {threshold:.4f}，证据相关性不足",
            )
        return RefusalDecision(False)

    if strategy == VECTOR:
        threshold = settings.retrieval_min_score if dense_min_score is None else dense_min_score
        top = max(candidate.score for candidate in candidates)
        if top < threshold:
            return RefusalDecision(
                True, SOURCE_RETRIEVAL,
                f"向量最高相似度 {top:.4f} 低于阈值 {threshold:.4f}，证据相关性不足",
            )
        return RefusalDecision(False)

    # 纯 BM25 与 hybrid（无精排）：分数分别是负数无界分与 RRF 名次倒数，都没有稳定量纲，
    # 不能拿来比阈值 —— 除空结果外不拒（spec §9.2 / 附录 A.3）。
    return RefusalDecision(False)


def self_check_gate(sufficient: bool, reason: str) -> RefusalDecision:
    if sufficient:
        return RefusalDecision(False)
    return RefusalDecision(True, SOURCE_SELF_CHECK, reason or "模型自评资料不足以回答")


def pool_record(raw_question: str, source: str, reason: str | None) -> dict:
    """落池记录，键与 `LowConfidenceRepository.add` 对齐。"""
    return {"raw_question": raw_question, "source": source, "reason": reason}
