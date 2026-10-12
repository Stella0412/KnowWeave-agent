"""建库材料清单：哪些文件、各算什么 content_type，只在此一处定义（CLI 与未来页面同读）。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    path: Path
    content_type: str


# 按文件名关键词匹配 content_type；匹配不到则退回 "md"（仍需切分，但算中性类型）
_KEYWORD_MAP = [
    ("policy", ("policy", "政策", "退货")),
    ("manual", ("manual", "手册", "售后")),
    ("faq", ("faq", "常见问题", "商品")),
]


def detect_content_type(filename: str) -> str:
    name = filename.lower()
    for content_type, keywords in _KEYWORD_MAP:
        if any(k in name for k in keywords):
            return content_type
    return "md"


def list_sources(docs_dir: str | Path) -> list[Source]:
    p = Path(docs_dir)
    if not p.is_dir():
        return []
    paths = [f for f in p.glob("*.md") if not f.name.startswith(".")]
    return [Source(path=f, content_type=detect_content_type(f.name)) for f in sorted(paths)]
