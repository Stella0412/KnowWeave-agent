"""Milvus 封装（schema v2）。默认 Milvus Lite（本地 .db，进程内 gRPC），MILVUS_URI 改成
http://127.0.0.1:19530 即切 standalone。

v2 相对 ch03 的变化（spec §5.1）：新增 `text`（开启分词器）与 `sparse`（由 BM25 函数自动生成），
并把过滤/引用要用的标量元数据（category / content_type / section_path / row_category /
is_key_clause）一起落进集合，让检索一次就拿全，不必回 MySQL。

⚠️ BM25 函数与分词器**只能在建集合时配置** → schema 变更必须 drop 重建（集合里是派生数据，重跑即得）。
⚠️ 本机已实测 Milvus Lite 支持全部这些能力（spec 附录 A.1）；官方文档在此自相矛盾，别被劝退。
"""

import logging
from pathlib import Path

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import Settings, settings

logger = logging.getLogger(__name__)

TEXT_MAX_LENGTH = 65535          # BM25 源文本 = category + questions + answer，取 VARCHAR 上限
QUESTION_MAX_LENGTH = 4096
ANSWER_MAX_LENGTH = 16384
META_MAX_LENGTH = 512
ROW_CATEGORY_MAX_LENGTH = 64

# ✅ 已核实（附录 A.2）：内置分词器只有 standard / jieba；文档里的 {"type": "chinese"} 在本机版本直接报错
_ANALYZER_PARAMS = {"tokenizer": "jieba", "filter": ["cnalphanumonly"]}

_OUTPUT_FIELDS = [
    "question", "answer", "category", "content_type", "section_path", "row_category",
]

_JIEBA_HINT = (
    "建集合失败：Milvus 的中文全文检索依赖 jieba 分词器。"
    "若报 unknown tokenizer type / JiebaAnalyzer 一类错误，先 `uv sync`（jieba 已在 pyproject.toml），"
    "再重试。"
)


class MilvusKnowledgeClient:
    def __init__(self, uri: str, collection: str, dim: int = 1024) -> None:
        self._client = MilvusClient(uri=uri)
        self._collection = collection
        self._dim = dim

    def ensure_collection(self) -> None:
        if self._client.has_collection(self._collection):
            # 集合已存在（另一个进程建的）时 load 状态是 NotLoad，本进程内 search 会报
            # released；load 是进程内的，故每次 attach 都要重新 load（幂等，已 load 再调无副作用）。
            self._client.load_collection(self._collection)
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field(
            "text", DataType.VARCHAR, max_length=TEXT_MAX_LENGTH,
            enable_analyzer=True, analyzer_params=_ANALYZER_PARAMS,
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("question", DataType.VARCHAR, max_length=QUESTION_MAX_LENGTH)
        schema.add_field("answer", DataType.VARCHAR, max_length=ANSWER_MAX_LENGTH)
        schema.add_field("category", DataType.VARCHAR, max_length=META_MAX_LENGTH)
        schema.add_field("content_type", DataType.VARCHAR, max_length=META_MAX_LENGTH)
        schema.add_field("section_path", DataType.VARCHAR, max_length=META_MAX_LENGTH)
        schema.add_field("row_category", DataType.VARCHAR, max_length=ROW_CATEGORY_MAX_LENGTH)
        schema.add_field("is_key_clause", DataType.BOOL)
        # sparse 由 BM25 函数从 text 自动生成，插入时**不喂值**；output_field_names 传字符串即可
        schema.add_function(
            Function(
                name="bm25_fn",
                input_field_names=["text"],
                output_field_names="sparse",
                function_type=FunctionType.BM25,
            )
        )
        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(
            field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
        )
        try:
            self._client.create_collection(
                collection_name=self._collection, schema=schema, index_params=index_params
            )
        except Exception as exc:
            logger.error("%s\n原始错误：%s", _JIEBA_HINT, exc)
            raise

    def drop_collection(self) -> None:
        """丢弃整张集合。schema v2 无法原地变更，重建的第一步就是它。"""
        if self._client.has_collection(self._collection):
            self._client.drop_collection(self._collection)

    def count(self) -> int:
        stats = self._client.get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))

    def upsert(self, rows: list[dict]) -> None:
        """整行 upsert。主键 = MySQL knowledge_chunks.id，重跑天然幂等。

        参数从 ch03 的四个位置参数改成行 dict：v2 要带的新字段太多，位置参数已经读不懂了。
        """
        if not rows:
            return
        self._client.upsert(collection_name=self._collection, data=rows)

    def search(self, vector: list[float], top_k: int, *, expr: str | None = None) -> list[dict]:
        """dense 单路。**不做阈值过滤** —— 阈值是拒答闸门的事（spec §9.2），
        评估召回率时提前过滤会把 Recall 算低。"""
        result = self._client.search(
            collection_name=self._collection,
            data=[vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
            **({"filter": expr} if expr else {}),
        )
        return _hits(result)

    def search_sparse(
        self, query_text: str, top_k: int, *, expr: str | None = None
    ) -> list[dict]:
        """BM25 单路：直接传原文字符串，集合的分析器负责分词编码（不手工喂稀疏向量）。

        `score` 是 BM25 距离，**为负数**（越负越相关），不可用于阈值判断。
        """
        result = self._client.search(
            collection_name=self._collection,
            data=[query_text],
            anns_field="sparse",
            search_params={"metric_type": "BM25"},
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
            **({"filter": expr} if expr else {}),
        )
        return _hits(result)

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_query_text: str,
        top_k: int,
        *,
        rrf_k: int = 60,
        expr: str | None = None,
    ) -> list[dict]:
        """dense ∥ BM25 两路，RRF 融合。`expr` 挂在**每一路**上（只过滤一路会漏成半过滤）。"""
        extra = {"expr": expr} if expr else {}
        reqs = [
            AnnSearchRequest(
                data=[dense_vector], anns_field="vector",
                param={"metric_type": "COSINE"}, limit=top_k, **extra,
            ),
            AnnSearchRequest(
                data=[sparse_query_text], anns_field="sparse",
                param={"metric_type": "BM25"}, limit=top_k, **extra,
            ),
        ]
        result = self._client.hybrid_search(
            collection_name=self._collection,
            reqs=reqs,
            ranker=RRFRanker(rrf_k),
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
        )
        return _hits(result)

    def query_by_expr(self, expr: str, *, limit: int = 16384) -> list[dict]:
        """按标量条件取原始行（不走向量检索）。评估集解析 ground-truth 用。"""
        return list(
            self._client.query(
                collection_name=self._collection,
                filter=expr,
                output_fields=["id"] + _OUTPUT_FIELDS,
                limit=limit,
            )
        )

    def delete_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        expr = "id in [" + ", ".join(str(int(i)) for i in ids) + "]"
        self._client.delete(collection_name=self._collection, filter=expr)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover - 关闭失败不影响进程退出
            logger.warning("关闭 Milvus 客户端失败", exc_info=True)


def _field(hit, name: str) -> str:
    """命中里取标量字段。字段没存过时 Milvus 会抛 KeyError，这里退化成空串。"""
    try:
        value = hit["entity"][name]
    except Exception:
        return ""
    return "" if value is None else str(value)


def _hits(result) -> list[dict]:
    return [
        {
            "id": int(hit["id"]),
            "score": float(hit["distance"]),
            "question": _field(hit, "question"),
            "answer": _field(hit, "answer"),
            "category": _field(hit, "category"),
            "content_type": _field(hit, "content_type"),
            "section_path": _field(hit, "section_path"),
            "row_category": _field(hit, "row_category"),
        }
        for hit in result[0]
    ]


def build_milvus_client(s: Settings | None = None) -> MilvusKnowledgeClient:
    s = s or settings
    uri = s.milvus_uri
    if not uri.startswith(("http://", "https://", "tcp://", "unix:")):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    return MilvusKnowledgeClient(uri=uri, collection=s.milvus_collection, dim=s.embed_dim)


_client: MilvusKnowledgeClient | None = None


def get_milvus_client() -> MilvusKnowledgeClient:
    """进程级懒加载单例：Milvus Lite 是进程内服务 + 独占锁，必须复用同一个客户端。"""
    global _client
    if _client is None:
        _client = build_milvus_client()
        _client.ensure_collection()
    return _client
