"""在线管线编排：改写 → 过滤 → 召回 → 融合 → 精排 → 装配 → 闸门。

`query_faq` 与评估脚本都调这一个 `run_pipeline` —— 评估测的必须就是线上那条链（spec §16）。

**同步函数**：整条链（embed / Milvus / rerank / LLM）都是阻塞 HTTP，调用方用
`asyncio.to_thread` 包一层；这样评估脚本可以直接调，不必先学会 async。

依赖注入（embed / milvus / reranker / model）全部可换：单测用假件，生产用单例。
"""

import logging
import time
from dataclasses import dataclass, field

from app.config import settings
from app.observability.context import get_observer
from app.observability.spans import observation_scope
from app.kb.sources import source_files_by_root
from app.rag.evidence import Evidence, assemble, build_evidence_block, to_citations
from app.rag.refusal import (
    SOURCE_RETRIEVAL,
    SOURCE_SELF_CHECK,
    RefusalDecision,
    retrieval_gate,
    self_check_gate,
)
from app.rag.rerank import RerankError, build_rerank_client
from app.rag.rewrite import RewriteResult, build_bm25_query, rewrite_query
from app.rag.strategies import embed_query, has_rerank, is_valid, recall

logger = logging.getLogger(__name__)

# 超长粘贴文本先截断再检索：BM25 分词与向量化都不该吃下整篇文档（Review Focus 4）
_QUERY_MAX_LENGTH = 2000


@dataclass
class TurnContext:
    """本轮会话的侧信道：用户原话 + 证据采集器 + 文档 H1→文件名反查表。

    由 ChatService 每轮建一个，传给 build_tools，query_faq 把证据写进来，
    流结束时 ChatService 取出推 citations 事件（spec §11）。
    """

    raw_question: str
    conversation_id: int | None = None
    citations: list[dict] = field(default_factory=list)
    source_files: dict[str, str] = field(default_factory=dict)


@dataclass
class RagResult:
    refused: bool
    raw_query: str
    strategy: str
    standard_query: str = ""
    refusal_source: str | None = None
    refusal_reason: str | None = None
    evidence_text: str = ""
    citations: list[dict] = field(default_factory=list)
    items: list[Evidence] = field(default_factory=list)
    rewrite_degraded: bool = False
    rerank_degraded: bool = False
    # 精排真的跑成时的最高分；None 表示未精排或精排降级（与拒答闸门的语义一致）。
    rerank_score: float | None = None


def _record_stage(kind, seconds, *, texts, items):
    observer = get_observer()
    if observer is None:
        return
    try:
        observer.record_stage(
            kind, model=(settings.embed_model if kind == "embedding"
                         else settings.rerank_model),
            duration_ms=int(seconds * 1000), texts=texts, items=items,
        )
    except Exception:
        logger.debug("阶段观测失败", exc_info=True)


def _refused(raw_query: str, strategy: str, decision: RefusalDecision, citations=None, **extra) -> RagResult:
    return RagResult(
        refused=True, raw_query=raw_query, strategy=strategy,
        refusal_source=decision.source, refusal_reason=decision.reason,
        citations=(citations if citations is not None else []), **extra,
    )


def _retrieval_summary(result: RagResult) -> dict:
    citations = result.citations or []
    return {
        "refused": result.refused,
        "refusal_source": result.refusal_source,
        "refusal_reason": result.refusal_reason,
        "document_count": len(citations),
        "documents": [
            {
                "n": item.get("n"),
                "chunk_id": item.get("chunk_id"),
                "section_path": item.get("section_path"),
                "score": item.get("score"),
            }
            for item in citations[:8]
        ],
        "rerank_score": result.rerank_score,
    }


def run_pipeline(
    raw_query: str,
    *,
    strategy: str | None = None,
    filters: dict[str, str | list[str]] | None = None,
    embed=None,
    milvus=None,
    reranker=None,
    model=None,
    source_files: dict[str, str] | None = None,
    recall_top_k: int | None = None,
    rerank_top_n: int | None = None,
    rrf_k: int | None = None,
    confidence_min_score: float | None = None,
    enable_self_check: bool = True,
) -> RagResult:
    """公开入口：包一层 retriever observation 后跑 `_run_pipeline`。

    `confidence_min_score` / `enable_self_check` 只给评估与校准用：把闸门放到 0 并跳过
    自评，才能在**不改变线上链路**的前提下拿到每一题的证据置信度。生产调用保持默认。
    """
    with observation_scope(
        "retriever",
        "retrieve-knowledge",
        input={
            "query": (raw_query or "").strip()[:_QUERY_MAX_LENGTH],
            "strategy": strategy or settings.rag_strategy,
        },
        metadata={
            "top_k": recall_top_k or settings.recall_top_k,
            "rerank_top_n": rerank_top_n or settings.rerank_top_n,
        },
    ) as span:
        result = _run_pipeline(
            raw_query,
            strategy=strategy,
            filters=filters,
            embed=embed,
            milvus=milvus,
            reranker=reranker,
            model=model,
            source_files=source_files,
            recall_top_k=recall_top_k,
            rerank_top_n=rerank_top_n,
            rrf_k=rrf_k,
            confidence_min_score=confidence_min_score,
            enable_self_check=enable_self_check,
        )
        if span is not None:
            span.update(output=_retrieval_summary(result))
        return result


def _run_pipeline(
    raw_query: str,
    *,
    strategy: str | None = None,
    filters: dict[str, str | list[str]] | None = None,
    embed=None,
    milvus=None,
    reranker=None,
    model=None,
    source_files: dict[str, str] | None = None,
    recall_top_k: int | None = None,
    rerank_top_n: int | None = None,
    rrf_k: int | None = None,
    confidence_min_score: float | None = None,
    enable_self_check: bool = True,
) -> RagResult:
    """跑一轮检索 + 质控，返回带引用的证据文本或拒答结论。

    抛 `MilvusError` 一类底层异常时**不吞**：那是「知识库不可用」，与「查不到依据」是两回事，
    由调用方（`query_faq`）映射成 UNAVAILABLE（Review Focus 1）。
    """
    query = (raw_query or "").strip()[:_QUERY_MAX_LENGTH]
    chosen = strategy or settings.rag_strategy
    if not is_valid(chosen):
        raise ValueError(f"未知检索策略：{chosen}")
    if not query:
        return _refused(
            query, chosen, RefusalDecision(True, SOURCE_RETRIEVAL, "问题为空，无法检索")
        )

    top_k = recall_top_k or settings.recall_top_k
    rrf = rrf_k or settings.rrf_k
    rerank_n = rerank_top_n or settings.rerank_top_n

    # 1. 改写（失败退回原话）
    rewrite = rewrite_query(query, model=model)
    bm25_query = build_bm25_query(query, rewrite)
    embed_text = rewrite.standard_query or query
    with observation_scope(
        "embedding", "embed-query",
        input={"query": embed_text}, model=settings.embed_model,
    ) as embedding:
        embed_start = time.perf_counter()
        dense_vector = embed_query(embed_text, embeddings=embed)
        _record_stage(
            "embedding", time.perf_counter() - embed_start,
            texts=[embed_text], items=1,
        )
        if embedding is not None:
            embedding.update(output={"dimensions": len(dense_vector or [])})

    # 2. 召回（过滤表达式挂在每一路）
    candidates = recall(
        chosen, dense_vector=dense_vector, bm25_query=bm25_query,
        top_k=top_k, rrf_k=rrf, filters=filters, milvus=milvus,
    )

    # 3. 精排（失败降级为不重排）
    rerank_degraded = False
    rerank_score: float | None = None
    items = candidates
    if has_rerank(chosen) and candidates:
        with observation_scope(
            "span", "rerank-evidence",
            input={"query": rewrite.standard_query or query,
                   "candidates": len(candidates), "top_n": rerank_n},
            metadata={"model": settings.rerank_model},
        ) as rerank_span:
            rerank_start = time.perf_counter()
            items, rerank_score, rerank_degraded = rerank_stage(
                candidates, rewrite.standard_query or query, reranker=reranker, top_n=rerank_n
            )
            _record_stage(
                "rerank", time.perf_counter() - rerank_start,
                texts=[rewrite.standard_query or query], items=len(items),
            )
            if rerank_span is not None:
                rerank_span.update(output={
                    "items": len(items), "top_score": rerank_score,
                    "degraded": rerank_degraded,
                })

    # 4. 检索侧早拒：置信度看「真正会进 prompt 的这几条」（精排后 items），不是召回全集
    decision = retrieval_gate(
        chosen, items, query=query, rerank_score=rerank_score,
        confidence_min_score=confidence_min_score,
    )
    if decision.refused:
        refusal_citations = (
            to_citations(assemble(items[:rerank_n]), source_files)
            if candidates and items else []
        )
        return _refused(
            query, chosen, decision, citations=refusal_citations,
            standard_query=rewrite.standard_query, rewrite_degraded=rewrite.degraded,
            rerank_degraded=rerank_degraded, rerank_score=rerank_score,
        )

    # 5. 首尾装配 + 编号
    assembled = assemble(items)
    evidence_text = build_evidence_block(assembled)
    citations = to_citations(assembled, source_files)

    # 6. 生成前自评（拿**用户原话**，不是改写后的问法；失败视为够答）
    if enable_self_check:
        sufficient, reason = _self_check(query, evidence_text, model=model)
        if sufficient is None:
            logger.warning("自评失败，按「够答」继续（不因一次抖动把能答的问题拒掉）")
        else:
            check = self_check_gate(sufficient, reason)
            if check.refused:
                return _refused(
                    query, chosen, check, citations=citations,
                    standard_query=rewrite.standard_query, rewrite_degraded=rewrite.degraded,
                    rerank_degraded=rerank_degraded, rerank_score=rerank_score,
                )

    return RagResult(
        refused=False, raw_query=query, strategy=chosen,
        standard_query=rewrite.standard_query, evidence_text=evidence_text,
        citations=citations, items=assembled, rewrite_degraded=rewrite.degraded,
        rerank_degraded=rerank_degraded, rerank_score=rerank_score,
    )


def run_expanded_pipeline(
    queries: list[str],
    resolved_query: str,
    *,
    strategy: str | None = None,
    filters: dict[str, str | list[str]] | None = None,
    embed=None,
    milvus=None,
    reranker=None,
    model=None,
    source_files: dict[str, str] | None = None,
    recall_top_k: int | None = None,
    rerank_top_n: int | None = None,
    rrf_k: int | None = None,
    confidence_min_score: float | None = None,
    enable_self_check: bool = True,
) -> RagResult:
    """公开入口：包一层 retriever observation 后跑 `_run_expanded_pipeline`。"""
    cleaned = [q.strip()[:_QUERY_MAX_LENGTH] for q in (queries or []) if q and q.strip()]
    with observation_scope(
        "retriever",
        "retrieve-policy-knowledge",
        input={
            "queries": cleaned,
            "resolved_query": (resolved_query or "").strip()[:_QUERY_MAX_LENGTH],
            "strategy": strategy or settings.rag_strategy,
        },
        metadata={
            "top_k": recall_top_k or settings.recall_top_k,
            "rerank_top_n": rerank_top_n or settings.rerank_top_n,
        },
    ) as span:
        result = _run_expanded_pipeline(
            queries,
            resolved_query,
            strategy=strategy,
            filters=filters,
            embed=embed,
            milvus=milvus,
            reranker=reranker,
            model=model,
            source_files=source_files,
            recall_top_k=recall_top_k,
            rerank_top_n=rerank_top_n,
            rrf_k=rrf_k,
            confidence_min_score=confidence_min_score,
            enable_self_check=enable_self_check,
        )
        if span is not None:
            span.update(output=_retrieval_summary(result))
        return result


def _run_expanded_pipeline(
    queries: list[str],
    resolved_query: str,
    *,
    strategy: str | None = None,
    filters: dict[str, str | list[str]] | None = None,
    embed=None,
    milvus=None,
    reranker=None,
    model=None,
    source_files: dict[str, str] | None = None,
    recall_top_k: int | None = None,
    rerank_top_n: int | None = None,
    rrf_k: int | None = None,
    confidence_min_score: float | None = None,
    enable_self_check: bool = True,
) -> RagResult:
    """多查询召回 → 按 chunk id 去重 → 用扩写主查询精排一次 → 单次自评。

    精排查询必须是**政策化**的扩写主查询（`cleaned[0]`），不能直接用口语原句：
    原句带着商品名/订单号（如「订单1001的猫爬架能退吗」），而政策条款里没有这些词，
    交叉编码器会把真正适用的条款压到证据置信度闸门以下（实测精排 0.23 vs 0.83，
    闸门见 `app/rag/confidence.py`）。
    扩写降级时 `cleaned == [resolved_query]`，行为自然退回单查询精排。
    """
    resolution = (resolved_query or "").strip()[:_QUERY_MAX_LENGTH]
    cleaned: list[str] = []
    for item in queries or []:
        q = (item or "").strip()[:_QUERY_MAX_LENGTH]
        if q and q not in cleaned:
            cleaned.append(q)
    if not cleaned:
        cleaned = [resolution]
    chosen = strategy or settings.rag_strategy
    if not is_valid(chosen):
        raise ValueError(f"未知检索策略：{chosen}")
    if not resolution:
        return _refused(
            resolution, chosen, RefusalDecision(True, SOURCE_RETRIEVAL, "问题为空，无法检索")
        )

    top_k = recall_top_k or settings.recall_top_k
    rrf = rrf_k or settings.rrf_k
    rerank_n = rerank_top_n or settings.rerank_top_n

    merged: dict[int, Evidence] = {}
    order: list[int] = []
    for query in cleaned:
        with observation_scope(
            "embedding", "embed-query",
            input={"query": query},
            metadata={"queries_in_batch": len(cleaned)},
            model=settings.embed_model,
        ) as embedding:
            embed_start = time.perf_counter()
            dense_vector = embed_query(query, embeddings=embed)
            _record_stage(
                "embedding", time.perf_counter() - embed_start,
                texts=[query], items=1,
            )
            if embedding is not None:
                embedding.update(output={"dimensions": len(dense_vector or [])})
        candidates = recall(
            chosen, dense_vector=dense_vector, bm25_query=query,
            top_k=top_k, rrf_k=rrf, filters=filters, milvus=milvus,
        )
        for candidate in candidates:
            previous = merged.get(candidate.id)
            if previous is None:
                merged[candidate.id] = candidate
                order.append(candidate.id)
            elif candidate.score > previous.score:
                merged[candidate.id] = candidate
    candidates = [merged[item_id] for item_id in order]

    rerank_degraded = False
    rerank_score: float | None = None
    items = candidates
    if has_rerank(chosen) and candidates:
        with observation_scope(
            "span", "rerank-evidence",
            input={"query": cleaned[0], "candidates": len(candidates),
                   "top_n": rerank_n},
            metadata={"model": settings.rerank_model},
        ) as rerank_span:
            rerank_start = time.perf_counter()
            items, rerank_score, rerank_degraded = rerank_stage(
                candidates, cleaned[0], reranker=reranker, top_n=rerank_n
            )
            _record_stage(
                "rerank", time.perf_counter() - rerank_start,
                texts=[cleaned[0]], items=len(items),
            )
            if rerank_span is not None:
                rerank_span.update(output={
                    "items": len(items), "top_score": rerank_score,
                    "degraded": rerank_degraded,
                })

    # 置信度看「真正会进 prompt 的这几条」（精排后 items），口径与主链路一致。
    decision = retrieval_gate(
        chosen, items, query=resolution, rerank_score=rerank_score,
        confidence_min_score=confidence_min_score,
    )
    if decision.refused:
        refusal_citations = (
            to_citations(assemble(items[:rerank_n]), source_files)
            if candidates and items else []
        )
        return _refused(
            resolution, chosen, decision, citations=refusal_citations,
            standard_query=resolution, rerank_degraded=rerank_degraded,
            rerank_score=rerank_score,
        )

    assembled = assemble(items)
    evidence_text = build_evidence_block(assembled)
    citations = to_citations(assembled, source_files)
    if enable_self_check:
        sufficient, reason = _self_check(resolution, evidence_text, model=model)
    else:
        sufficient = None
    if sufficient is not None:
        check = self_check_gate(sufficient, reason)
        if check.refused:
            return _refused(
                resolution, chosen, check, citations=citations,
                standard_query=resolution, rerank_degraded=rerank_degraded,
                rerank_score=rerank_score,
            )
    return RagResult(
        refused=False, raw_query=resolution, strategy=chosen,
        standard_query=resolution, evidence_text=evidence_text,
        citations=citations, items=assembled,
        rerank_degraded=rerank_degraded, rerank_score=rerank_score,
    )


def rerank_stage(
    candidates: list[Evidence],
    query: str,
    *,
    reranker=None,
    top_n: int | None = None,
) -> tuple[list[Evidence], float | None, bool]:
    """精排一段候选：返回 `(进 prompt 的证据, 精排最高分, 是否降级)`。

    `run_pipeline` 与检索段评估脚本**共用这一个函数** —— 评估要测的必须就是线上那条链
    （spec §16）。谁都不许另写一份等价实现：复制品会在管线变更时静默漂移。

    调用方只在 `candidates` 非空时调它（`run_pipeline` 的
    `if has_rerank(chosen) and candidates:` 里那个 `and` 就是干这个的）。
    """
    rerank_n = top_n or settings.rerank_top_n
    try:
        client = reranker or build_rerank_client()
        ranked = client.rerank(
            query, [candidate.rerank_text for candidate in candidates], top_n=rerank_n
        )
        items, rerank_score = _apply_rerank(candidates, ranked)
        # 精排没真的生效（上游返回空表 / 下标全非法）同样是「降级」：证据条数必须与
        # 下面 RerankError 那条路径**一致**（都截到 rerank_n）。`_apply_rerank` 在
        # picked 为空时返回的是**原始候选全集**（recall_top_k 默认 50），不截的话
        # 同一个「精排没生效」情形会往 prompt 里塞 50 条、引用面板推 50 条角标，
        # 而另一条降级路径只塞 10 条 —— 两条路给两个答案。
        # 拒答判定不受影响：rerank_score 仍是 None（Ruling R1）。
        if rerank_score is None:
            items = candidates[:rerank_n]
        return items, rerank_score, rerank_score is None
    except RerankError:
        logger.warning("精排不可用，降级为 RRF 顺序", exc_info=True)
        return candidates[:rerank_n], None, True


def _apply_rerank(
    candidates: list[Evidence], ranked: list[tuple[int, float]]
) -> tuple[list[Evidence], float | None]:
    """按精排结果重排候选：`ranked` 是 [(原候选下标, 相关性分)]，分即 Evidence.score。

    返回 `(进 prompt 的证据, 精排最高分)`，**两个值必须一起给**（Ruling R1）：
    调用方一旦只拿到证据，就只能从 items 的 score 反推精排分，而 items 在兜底路径上
    就是原始 candidates —— 那批 score 是 hybrid_search 的 RRF 名次倒数（≈0.016~0.033），
    混进精排闸门会把每一道题都拒掉。所以**精排没真的生效时（`ranked` 空表、或下标全非法），
    精排分必须返回 None**，让闸门知道「这是降级，别比精排阈值」。
    """
    picked = []
    for index, score in ranked:
        if 0 <= index < len(candidates):
            candidate = candidates[index]
            candidate.score = score      # 生产策略下 score 的语义 = 精排相关性分
            picked.append(candidate)
    if not picked:
        return candidates, None
    return picked, max(candidate.score for candidate in picked)


def _self_check(
    raw_query: str, evidence_text: str, *, model=None
) -> tuple[bool | None, str]:
    """返回 (sufficient, reason)；`sufficient=None` 表示自评没跑成。"""
    from app.core.llm import build_structured_model
    from app.prompts.rag import SelfCheckOutput, self_check_messages

    try:
        # 自评是拒答闸门，temperature=0：同一题同一证据必须给出同一个结论。
        checker = model or build_structured_model(settings, SelfCheckOutput, temperature=0.0)
        output = checker.invoke(self_check_messages(raw_query, evidence_text))
    except Exception:
        logger.warning("生成前自评失败", exc_info=True)
        return None, ""
    return bool(getattr(output, "sufficient", False)), str(getattr(output, "reason", "") or "")


def default_source_files() -> dict[str, str]:
    """文档 H1 → 文件名，用来给引用面板反查「查看源文档」入口。取不到就返回空表。"""
    try:
        return source_files_by_root(settings.kb_docs_dir)
    except Exception:
        logger.warning("读取 KB 文件清单失败，引用面板不给源文件入口", exc_info=True)
        return {}
