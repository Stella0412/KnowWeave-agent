"""四种检索策略的召回段 + 标量过滤表达式拼装。

四策略共用同一条管线、同一份过滤与装配逻辑，只有「跑哪几路 / 排不排序」不同 ——
这是对比报告可信的前提（spec §7.1）。

**召回段不做阈值过滤**：阈值是拒答闸门的事（spec §9.2）。在召回段提前按分数丢结果，
会把 recall 算低，评估出来的数字就不再是「这条链能召回什么」。
"""

from app.core.embeddings import embed_texts
from app.kb.milvus_client import MilvusKnowledgeClient, get_milvus_client
from app.rag.evidence import Evidence, evidence_from_hit

VECTOR = "vector"
BM25 = "bm25"
HYBRID = "hybrid"
HYBRID_RERANK = "hybrid_rerank"

STRATEGIES: tuple[str, ...] = (VECTOR, BM25, HYBRID, HYBRID_RERANK)

# 允许出现在过滤表达式里的字段。值来自白名单枚举而不是用户自由文本（spec §8）
FILTER_FIELDS: tuple[str, ...] = ("row_category", "content_type", "category")

_RERANK_MARK = "rerank"


def is_valid(strategy: str) -> bool:
    return strategy in STRATEGIES


def has_rerank(strategy: str) -> bool:
    return _RERANK_MARK in strategy


def _quote(value: str) -> str:
    """Milvus 表达式的字符串字面量用双引号；反斜杠与引号都要转义。

    值本该来自白名单枚举，但评估集标注、工具入参都可能带引号，拼串会把整条表达式搞成语法错。
    """
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_expr(filters: dict[str, str | list[str]] | None) -> str | None:
    """filters → Milvus 标量表达式。空 / None 表示不过滤（全库）。"""
    if not filters:
        return None
    clauses = []
    for field, value in filters.items():
        if field not in FILTER_FIELDS:
            raise ValueError(f"不支持的过滤字段：{field}（可选：{'/'.join(FILTER_FIELDS)}）")
        if isinstance(value, (list, tuple, set)):
            values = [_quote(v) for v in value]
            if not values:
                continue
            clauses.append(f"{field} in [{', '.join(values)}]")
        else:
            clauses.append(f"{field} == {_quote(value)}")
    if not clauses:
        return None
    return " and ".join(clauses)


def embed_query(query: str, *, embeddings=None) -> list[float]:
    return embed_texts([query], embeddings=embeddings)[0]


def recall(
    strategy: str,
    *,
    dense_vector: list[float],
    bm25_query: str,
    top_k: int,
    rrf_k: int,
    filters: dict[str, str | list[str]] | None = None,
    milvus: MilvusKnowledgeClient | None = None,
) -> list[Evidence]:
    """按策略跑召回，返回 Evidence 列表（顺序即该策略的候选顺序）。"""
    if not is_valid(strategy):
        raise ValueError(f"未知检索策略：{strategy}（可选：{'/'.join(STRATEGIES)}）")
    store = milvus or get_milvus_client()
    expr = build_expr(filters)

    if strategy == VECTOR:
        hits = store.search(dense_vector, top_k, expr=expr)
    elif strategy == BM25:
        hits = store.search_sparse(bm25_query, top_k, expr=expr)
    else:
        hits = store.hybrid_search(
            dense_vector, bm25_query, top_k, rrf_k=rrf_k, expr=expr
        )
    return [evidence_from_hit(hit) for hit in hits]
