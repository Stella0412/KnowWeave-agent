"""Query 理解：一次 LLM 调用产出「标准问法 + 同义词」。

标准问法走 dense 路（治口语问法的向量偏移），同义词拼进 BM25 路（治字面不匹配）。
**入库侧不动**：知识库仍是「一条知识一份文本」，同义词只在检索时展开（用户明确要求）。

BGE 之外的一个实测事实（附录 A.3）：BM25 多词查询不要求文档含全部词，是 OR 式打分，
所以同义词直接拼进查询串即可，不必拼 TEXT_MATCH 表达式。
"""

import logging
from dataclasses import dataclass, field

from app.config import Settings, settings
from app.core.llm import build_structured_model
from app.prompts.rag import RewriteOutput, rewrite_messages

logger = logging.getLogger(__name__)

_MAX_SYNONYMS = 5


@dataclass(frozen=True)
class RewriteResult:
    standard_query: str
    synonyms: list[str] = field(default_factory=list)
    # True = 改写没跑成，standard_query 就是原话。评估报告里要看得出这一轮降级过
    degraded: bool = False


_model = None


def build_rewrite_model(s: Settings | None = None):
    """进程级懒加载：每次检索都重建结构化模型会白扔一个 httpx 客户端。"""
    global _model
    if _model is None:
        _model = build_structured_model(s or settings, RewriteOutput)
    return _model


def rewrite_query(raw_query: str, *, model=None) -> RewriteResult:
    """把口语问法归一成标准问法 + 同义词。任何失败都退回原话，不抛。"""
    raw = (raw_query or "").strip()
    if not raw:
        return RewriteResult(standard_query="", degraded=True)
    try:
        output = (model or build_rewrite_model()).invoke(rewrite_messages(raw))
    except Exception:
        logger.warning("Query 改写失败，退回原话检索", exc_info=True)
        return RewriteResult(standard_query=raw, degraded=True)

    standard = (getattr(output, "standard_query", "") or "").strip()
    if not standard:
        return RewriteResult(standard_query=raw, degraded=True)
    return RewriteResult(
        standard_query=standard,
        synonyms=_clean_synonyms(getattr(output, "synonyms", None) or [], standard),
        degraded=False,
    )


def _clean_synonyms(synonyms: list[str], standard_query: str) -> list[str]:
    """去空白、去重、去掉与标准问法重复的、限长。模型偶尔会吐重复项或空串。"""
    cleaned: list[str] = []
    seen = {standard_query}
    for item in synonyms:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
        if len(cleaned) >= _MAX_SYNONYMS:
            break
    return cleaned


def build_bm25_query(raw_query: str, rewrite: RewriteResult) -> str:
    """拼 BM25 查询串：原话 + 标准问法 + 同义词，按空格分隔并去重。

    保留原话是因为改写可能丢词；三者并列靠 BM25 的 OR 式打分各自贡献权重。
    """
    parts: list[str] = []
    for text in [raw_query, rewrite.standard_query, *rewrite.synonyms]:
        value = (text or "").strip()
        if value and value not in parts:
            parts.append(value)
    return " ".join(parts)
