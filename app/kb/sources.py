"""建库材料清单：哪些文件、各算什么 content_type，只在此一处定义（CLI 与未来页面同读）。"""

from dataclasses import dataclass
from pathlib import Path

from app.kb.chunking import first_heading


@dataclass(frozen=True)
class Source:
    path: Path
    content_type: str


# 按文件名关键词匹配 content_type；匹配不到则退回 "md"（仍需切分，但算中性类型）
# ⚠️ 关键词表必须同时认中文名与英文名：夹具换成英文命名后，只认中文名会让 3 篇静默退化成 md
# （content_type 是 §8 的过滤维度之一，退化不报错、只悄悄把过滤能力砍半）。
_KEYWORD_MAP = [
    ("policy", ("policy", "政策", "退货", "returns", "billing", "shipping", "member", "benefit")),
    ("manual", ("manual", "手册", "售后", "after-sales")),
    ("faq", ("faq", "常见问题")),
    ("spec", ("spec", "规格", "型号")),
]


def detect_content_type(filename: str) -> str:
    name = filename.lower()
    for content_type, keywords in _KEYWORD_MAP:
        if any(k in name for k in keywords):
            return content_type
    return "md"


def source_files_by_root(docs_dir: str | Path) -> dict[str, str]:
    """文档 H1 标题 → 文件名。原始文件名没有落库，前端要「查看源文档」时靠这张表反查。

    查不到（KB 目录不可见、或文档无标题）就不给入口，不硬编文件名。
    """
    mapping: dict[str, str] = {}
    for source in list_sources(docs_dir):
        try:
            root = first_heading(source.path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if root:
            mapping[root] = source.path.name
    return mapping


def list_sources(docs_dir: str | Path) -> list[Source]:
    p = Path(docs_dir)
    if not p.is_dir():
        return []
    paths = [f for f in p.glob("*.md") if not f.name.startswith(".")]
    return [Source(path=f, content_type=detect_content_type(f.name)) for f in sorted(paths)]
