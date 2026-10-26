from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "config" / "ch10_taxonomy.json"


class TaxonomyError(ValueError):
    pass


@dataclass(frozen=True)
class Category:
    name: str
    definition: str
    boundary: str
    keywords: tuple[str, ...]
    examples: tuple[str, ...]
    confusable_with: tuple[str, ...]
    positive_cues: tuple[str, ...]
    negative_cues: tuple[str, ...]


@dataclass(frozen=True)
class Taxonomy:
    version: str
    categories: tuple[Category, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.categories)

    @property
    def other(self) -> str:
        return self.names[-1]

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise TaxonomyError(f"未知标签: {name}") from exc

    def multi_hot(self, labels: Iterable[str]) -> list[float]:
        vec = [0.0] * len(self.categories)
        for name in labels:
            vec[self.index(name)] = 1.0
        return vec

    def ordered(self, labels: Iterable[str]) -> list[str]:
        wanted = set(labels)
        for name in wanted:
            self.index(name)
        return [name for name in self.names if name in wanted]

    def prompt_table(self) -> str:
        lines = []
        for i, c in enumerate(self.categories, 1):
            lines.append(f"{i}. {c.name}：{c.definition} 边界：{c.boundary} 易混：{','.join(c.confusable_with) or '无'}")
        lines.append("近邻边界：修归保修维修、退归退换货；运费管钱、物流管货；价保是补差价、优惠活动是券和满减。")
        return "\n".join(lines)


def _require(category: dict, key: str) -> str:
    value = str(category.get(key) or "").strip()
    if not value:
        raise TaxonomyError(f"类目 {category.get('name')} 缺少 {key}")
    return value


def load_taxonomy(path: str | Path = DEFAULT_TAXONOMY_PATH) -> Taxonomy:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaxonomyError(f"类目表不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TaxonomyError(f"类目表不是合法 JSON: {exc}") from exc

    items = raw.get("labels")
    if not isinstance(items, list) or len(items) != 17:
        raise TaxonomyError("类目表必须正好 17 项")
    categories = []
    for item in items:
        name = _require(item, "name")
        categories.append(Category(
            name=name,
            definition=_require(item, "definition"),
            boundary=_require(item, "boundary"),
            keywords=tuple(item.get("keywords") or ()),
            examples=tuple(item.get("examples") or ()),
            confusable_with=tuple(item.get("confusable_with") or ()),
            positive_cues=tuple(item.get("positive_cues") or ()),
            negative_cues=tuple(item.get("negative_cues") or ()),
        ))
    names = [c.name for c in categories]
    if len(set(names)) != 17:
        raise TaxonomyError("类目名必须唯一")
    if names[-1] != "其他":
        raise TaxonomyError("『其他』必须固定最后")
    for c in categories:
        for other in c.confusable_with:
            if other not in names:
                raise TaxonomyError(f"{c.name} 的易混类 {other} 不在类目表")
    return Taxonomy(version=str(raw.get("version") or "unknown"), categories=tuple(categories))
