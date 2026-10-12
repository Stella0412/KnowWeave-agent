"""Milvus 封装。默认 Milvus Lite（本地 .db，进程内 gRPC），MILVUS_URI 改成
http://127.0.0.1:19530 即切 standalone。集合 knowledge 存 {id, vector, question, answer}，
主键 = MySQL knowledge_chunks.id（不用 auto_id，重跑 upsert 天然幂等）。
"""

import logging
from pathlib import Path

from pymilvus import DataType, MilvusClient
from pymilvus.milvus_client import IndexParams

from app.config import Settings, settings

logger = logging.getLogger(__name__)

QUESTION_MAX_LENGTH = 4096
ANSWER_MAX_LENGTH = 16384


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
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("question", DataType.VARCHAR, max_length=QUESTION_MAX_LENGTH)
        schema.add_field("answer", DataType.VARCHAR, max_length=ANSWER_MAX_LENGTH)
        index_params = IndexParams()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        self._client.create_collection(
            collection_name=self._collection, schema=schema, index_params=index_params
        )

    def count(self) -> int:
        stats = self._client.get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))

    def upsert(
        self,
        ids: list[int],
        vectors: list[list[float]],
        questions: list[str],
        answers: list[str],
    ) -> None:
        if not ids:
            return
        rows = [
            {"id": int(i), "vector": v, "question": q, "answer": a}
            for i, v, q, a in zip(ids, vectors, questions, answers)
        ]
        self._client.upsert(collection_name=self._collection, data=rows)

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        result = self._client.search(
            collection_name=self._collection,
            data=[vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=["question", "answer"],
        )
        return [
            {
                "id": int(hit["id"]),
                "question": hit["entity"]["question"],
                "answer": hit["entity"]["answer"],
                "score": float(hit["distance"]),
            }
            for hit in result[0]
        ]

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
