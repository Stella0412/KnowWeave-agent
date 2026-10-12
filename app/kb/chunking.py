"""Markdown 结构感知切分。

标题层级切用 LangChain 的 MarkdownHeaderTextSplitter（spec §6 指定）；超长段递归切
（RecursiveCharacterTextSplitter）、大表格按行分组复制表头、重叠裁到句末、问法别名与
关键条款标记，分别在 Task 5 / Task 6 作为薄层补上。
"""

import re
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter

_ENC = tiktoken.get_encoding("cl100k_base")  # 与 ch02 ContextStore 同口径（近似计数）

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_ALIAS_RE = re.compile(r"^问法\s*[:：]\s*(.+)$")
KEY_CLAUSE_MARK = "【关键条款】"

# 标题层级 → MarkdownHeaderTextSplitter 的 metadata 键名（与 headers_to_split_on 一一对应）
_HEADERS_TO_SPLIT = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]


@dataclass(frozen=True)
class ChunkLimits:
    target_tokens: int = 500
    max_tokens: int = 800
    overlap_tokens: int = 80
    table_max_rows: int = 30


@dataclass(frozen=True)
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str | None = None
    content_type: str = "md"
    is_key_clause: bool = False


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def first_heading(text: str) -> str | None:
    """返回文档第一个标题的文本（通常是 H1），用作文档级幂等根；无标题返回 None。"""
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            return match.group(2)
    return None


def _header_path(metadata: dict) -> list[str]:
    """把 MarkdownHeaderTextSplitter 的 metadata 还原成从根到自身的标题路径。"""
    path = []
    for i in range(1, 7):
        key = f"Header {i}"
        if key in metadata:
            path.append(metadata[key])
    return path


def _build_header_splitter() -> MarkdownHeaderTextSplitter:
    return MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS_TO_SPLIT, strip_headers=True)


def split_markdown(
    text: str, *, default_content_type: str = "md", limits: ChunkLimits | None = None
) -> list[Chunk]:
    """切分 Markdown 为 Chunk 列表。标题层级切交给 MarkdownHeaderTextSplitter。"""
    limits = limits or ChunkLimits()
    chunks: list[Chunk] = []
    for doc in _build_header_splitter().split_text(text):
        path = _header_path(doc.metadata)
        if not path:  # 没有标题的正文不产出 chunk（空文档守卫）
            continue
        body = doc.page_content.strip()
        if not body:  # 空章节跳过
            continue
        questions, body = _extract_questions_and_aliases(body, path[-1], default_content_type)
        category = " > ".join(path[:-1]) if len(path) > 1 else path[0]
        section_path = " > ".join(path)
        for piece in _split_long_text(body, limits):
            chunks.append(
                Chunk(
                    category=category[:255],
                    questions=questions[:65535],
                    answer=piece,
                    section_path=section_path[:512],
                    content_type=default_content_type,
                    is_key_clause=KEY_CLAUSE_MARK in piece,
                )
            )
    return chunks


def _extract_questions_and_aliases(
    body: str, title: str, content_type: str
) -> tuple[str, str]:
    """返回 (questions, clean_body)。FAQ 类从正文抽 `问法：a / b` 行作为别名；其余类型直接用标题当问法。"""
    aliases: list[str] = []
    lines: list[str] = []
    for line in body.splitlines():
        match = _ALIAS_RE.match(line.strip()) if content_type == "faq" else None
        if match:
            aliases.extend(p.strip() for p in re.split(r"[/、,，]", match.group(1)) if p.strip())
        else:
            lines.append(line)
    clean_body = "\n".join(lines).strip()
    questions = title
    if aliases:
        questions = title + "\n" + "\n".join(aliases)
    return questions, clean_body


def _split_long_text(body: str, limits: ChunkLimits) -> list[str]:
    """最小实现：正文整体作为一块。Task 5 扩展为「超长段 RecursiveCharacterTextSplitter 递归切 + 重叠裁句」。"""
    return [body]
