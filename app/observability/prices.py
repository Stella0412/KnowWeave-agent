"""ch09 单价配置：null 表示未配价，成本记 0，不伪造金额。"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class PriceBook:
    prices: dict[str, ModelPrice] = field(default_factory=dict)

    def get(self, model: str) -> ModelPrice | None:
        return self.prices.get(model)

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = self.get(model)
        if price is None:
            return 0.0
        return round(
            prompt_tokens / 1_000_000 * price.input_per_million
            + completion_tokens / 1_000_000 * price.output_per_million,
            6,
        )


def load_prices(path: str | Path) -> PriceBook:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        logger.warning("读取模型单价失败，按全部未配价处理", exc_info=True)
        return PriceBook({})
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict):
        logger.warning("模型单价文件缺少 models 对象，按空配置处理")
        return PriceBook({})
    prices: dict[str, ModelPrice] = {}
    for name, item in models.items():
        if not isinstance(item, dict):
            continue
        inp = item.get("input_per_million")
        out = item.get("output_per_million")
        if inp is None or out is None:
            continue
        try:
            inp_f, out_f = float(inp), float(out)
        except (TypeError, ValueError):
            logger.warning("模型 %s 单价不是数字，已忽略", name)
            continue
        if inp_f < 0 or out_f < 0:
            logger.warning("模型 %s 单价为负数，已忽略", name)
            continue
        prices[name] = ModelPrice(inp_f, out_f)
    return PriceBook(prices)
