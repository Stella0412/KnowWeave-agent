"""证据对象、首尾装配、引用编号与 prompt 证据块。

两件事在这里定死，别处不许再各写一遍：
1. **装配序**：精排后是「最相关 → 最不相关」，组装进 prompt 时按首尾优先重排
   （对抗长上下文里的「中间遗忘」）；
2. **编号按装配序赋**（不是按精排分序）—— 编号是给模型看的，prompt 里模型先读到的那条
   必须是 [1]，否则模型「引用第 1 条」和它实际先读到的东西不一致。
"""

from dataclasses import dataclass

_CONTENT_PREVIEW_MAX = 4000


@dataclass
class Evidence:
    id: int
    question: str
    answer: str
    section_path: str
    content_type: str
    row_category: str
    category: str
    score: float
    n: int = 0

    @property
    def doc_root(self) -> str:
        """文档根 = section_path 的第一段（即文档 H1）。原始文件名没有落库，只能从这派生。"""
        return self.section_path.split(" > ")[0].strip() if self.section_path else ""

    @property
    def rerank_text(self) -> str:
        """喂给精排的文本：问题（含小节标题与问法别名）+ 正文。"""
        return f"{self.question}\n{self.answer}".strip()


def evidence_from_hit(hit: dict) -> Evidence:
    return Evidence(
        id=int(hit.get("id", 0)),
        question=hit.get("question") or "",
        answer=hit.get("answer") or "",
        section_path=hit.get("section_path") or "",
        content_type=hit.get("content_type") or "",
        row_category=hit.get("row_category") or "",
        category=hit.get("category") or "",
        score=float(hit.get("score", 0.0)),
    )


def head_tail_order(count: int) -> list[int]:
    """精排序下标 → 装配序下标。取奇数位放前、偶数位倒序放后：

    count=10 → [0, 2, 4, 6, 8, 9, 7, 5, 3, 1]（与 spec §7.4 一致）。
    """
    if count <= 0:
        return []
    indices = list(range(count))
    return indices[0::2] + indices[1::2][::-1]


def assemble(items: list[Evidence]) -> list[Evidence]:
    """按首尾序重排并**按装配序**赋编号（1 起）。返回新列表，不改输入元素的分值。"""
    ordered = [
        Evidence(
            id=item.id, question=item.question, answer=item.answer,
            section_path=item.section_path, content_type=item.content_type,
            row_category=item.row_category, category=item.category,
            score=item.score, n=index + 1,
        )
        for index, item in enumerate([items[i] for i in head_tail_order(len(items))])
    ]
    return ordered


def build_evidence_block(items: list[Evidence]) -> str:
    """拼进 prompt 的证据块。每条以 [n] 开头，并带上章节路径便于模型按需引用。"""
    blocks = []
    for item in items:
        header = f"[{item.n}] {item.section_path}" if item.section_path else f"[{item.n}]"
        if item.row_category:
            header += f"（分类：{item.row_category}）"
        blocks.append(f"{header}\n问：{item.question}\n答：{item.answer}")
    return "\n\n".join(blocks)


def to_citations(items: list[Evidence], source_files: dict[str, str] | None = None) -> list[dict]:
    """SSE `citations` 事件载荷（spec §12）。

    `source_file` 是后端加的一列（⚠️ spec 未覆盖，spec 只说「能反查到文件时给入口」）：
    原始文件名没有落库，只有后端能反查，前端拿不到；查不到就给 None，前端不显示那个入口。
    `score` 在不同策略下量纲不同（精排分 / 余弦 / BM25 负数），前端只展示，不据此判断。
    """
    mapping = source_files or {}
    return [
        {
            "n": item.n,
            "chunk_id": item.id,
            "doc_root": item.doc_root,
            "source_file": mapping.get(item.doc_root),
            "section_path": item.section_path,
            "question": item.question,
            "answer": item.answer[:_CONTENT_PREVIEW_MAX],
            "content_type": item.content_type,
            "row_category": item.row_category or None,
            "score": item.score,
        }
        for item in items
    ]
