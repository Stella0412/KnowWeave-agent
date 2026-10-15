"""交叉编码器精排：硅基流动 `POST /v1/rerank`，模型 bge-reranker-v2-m3。

用 urllib 而不是 httpx：httpx 只是 dev 依赖，rerank 是**生产路径**，不该为一个 POST
把它提成运行时依赖（附录 A.4 的实测探针也是这么调的）。

失败一律抛 RerankError：调用方对「上游挂了」只做一件事 —— 降级为不重排（spec §7.3）。
不要在这里吞异常，否则上层分不清「没排」和「排了但分都很低」。
"""

import json
import logging
import urllib.error
import urllib.request

from app.config import Settings, settings

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """重排上游不可用 / 返回体不合预期。调用方据此降级为不重排。"""


class RerankClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def is_configured(self) -> bool:
        return bool(self._base_url and self._api_key and self._model)

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """返回 [(原候选下标, 相关性分)]，按分降序。空候选直接返回空（不打上游）。"""
        if not documents:
            return []
        if not self.is_configured():
            raise RerankError("重排未配置：缺 base_url / api_key / model")

        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": max(1, min(top_n, len(documents))),
        }
        request = urllib.request.Request(
            self._base_url + "/rerank",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # 只记状态码，不记 body —— 有些网关会把请求头（含 Authorization）回显在错误里
            raise RerankError(f"重排上游返回 HTTP {exc.code}") from exc
        except Exception as exc:
            raise RerankError(f"重排上游调用失败：{type(exc).__name__}") from exc

        try:
            pairs = [(int(item["index"]), float(item["relevance_score"])) for item in body["results"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankError("重排上游返回体不合预期") from exc
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)


def build_rerank_client(s: Settings | None = None) -> RerankClient:
    """base_url / api_key 留空即复用嵌入那套端点与密钥（同一个硅基流动账号）。"""
    s = s or settings
    return RerankClient(
        base_url=s.rerank_base_url or s.embed_base_url,
        api_key=s.rerank_api_key or s.embed_api_key,
        model=s.rerank_model,
        timeout=s.rerank_timeout,
    )
