"""Markdown 结构感知切分。

标题层级切用 LangChain 的 MarkdownHeaderTextSplitter（spec §6 指定）；超长段递归切
（RecursiveCharacterTextSplitter）、大表格按行分组复制表头、重叠裁到句末、问法别名与
关键条款标记，分别在 Task 5 / Task 6 作为薄层补上。
"""

import re
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_ENC = tiktoken.get_encoding("cl100k_base")  # 与 ch02 ContextStore 同口径（近似计数）

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_ALIAS_RE = re.compile(r"^问法\s*[:：]\s*(.+)$")
KEY_CLAUSE_MARK = "【关键条款】"

# 超长段递归切分的分隔符优先级：段落 → 换行 → 中文句末 → 英文句末 → 空白 → 字符兜底（末尾 "" 触发字符级硬切）
_LONG_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]

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


def trim_to_sentence_end(text: str) -> str:
    """把文本末尾裁到最近的句末符（含）。没有句末符则原样返回。"""
    stripped = text.rstrip()
    for i in range(len(stripped) - 1, -1, -1):
        if stripped[i] in "。！？；!?;":
            return stripped[: i + 1]
    return stripped


def _long_splitter(limits: ChunkLimits) -> RecursiveCharacterTextSplitter:
    """超长段的递归细切器：chunk_size 用目标 token 数；重叠统一交给 _apply_overlap 裁句（chunk_overlap=0 避免双重重叠）。

    `keep_separator="end"`：让句末符留在**前一块**末尾（默认 True 等价 "start"，会把「。」贴到下一块开头，块尾就留半截话）。
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=max(1, limits.target_tokens),
        chunk_overlap=0,
        keep_separator="end",
        separators=_LONG_SEPARATORS,
    )


def _split_long_text(body: str, limits: ChunkLimits) -> list[str]:
    """超长正文 → 多块：表格按行切并复制表头；文本走 RecursiveCharacterTextSplitter；短内容原样一块。"""
    if count_tokens(body) <= limits.max_tokens:
        return [body]

    pieces: list[str] = []
    for kind, block_lines in split_blocks(body.splitlines()):
        if kind == "table":
            pieces.extend(_split_table(block_lines, limits))
        else:
            pieces.extend(_long_splitter(limits).split_text("\n".join(block_lines)))
    return _apply_overlap(pieces, limits.overlap_tokens)


def is_table_block(lines: list[str]) -> bool:
    """连续 | 行（含 --- 分隔行）构成表格块。要求至少两行且含分隔行，避免把零星竖线误判。"""
    if len(lines) < 2:
        return False
    if not all(line.strip().startswith("|") for line in lines):
        return False
    return any(set(line.strip()) <= set("|-: ") and "-" in line for line in lines[1:])


def split_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    """把行序列切成 ("text"|"table", lines) 块序列。"""
    blocks: list[tuple[str, list[str]]] = []
    buffer: list[str] = []
    kind = "text"

    def flush() -> None:
        if buffer:
            blocks.append((kind, list(buffer)))
            buffer.clear()

    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            if is_table_block(lines[i:j]):
                flush()
                blocks.append(("table", lines[i:j]))
                i = j
                continue
        buffer.append(lines[i])
        i += 1
    flush()
    return blocks


def _split_table(lines: list[str], limits: ChunkLimits) -> list[str]:
    """大表格按行切，每片复制表头 + 分隔行；每片行数受 table_max_rows 限制，并按 max_tokens 收缩。

    逐片推进（而非递归对半砍），保证每轮至少吃掉一行 → 绝不会不收敛（单行本身超 max_tokens 时，
    该行独占一片、接受超限，好过死循环）。
    """
    if len(lines) <= limits.table_max_rows + 2:
        return ["\n".join(lines) + "\n"]

    prefix = f"{lines[0].strip()}\n{lines[1].strip()}\n"
    rows = lines[2:]

    groups: list[str] = []
    i = 0
    while i < len(rows):
        take = max(1, limits.table_max_rows)
        while take > 1:
            candidate = prefix + "".join(row.strip() + "\n" for row in rows[i : i + take])
            if count_tokens(candidate) <= limits.max_tokens:
                break
            take //= 2
        chunk_rows = rows[i : i + take]
        groups.append(prefix + "".join(row.strip() + "\n" for row in chunk_rows))
        i += len(chunk_rows)
    return groups


def _apply_overlap(pieces: list[str], overlap_tokens: int) -> list[str]:
    """给第 2 块起的每块头部接上前一块尾部 overlap_tokens，并把接缝对齐到句首。"""
    if overlap_tokens <= 0 or len(pieces) < 2:
        return pieces
    result = [pieces[0]]
    for prev, piece in zip(pieces, pieces[1:]):
        overlap = _align_to_sentence_start(_tail_tokens(prev, overlap_tokens))
        result.append((overlap + piece) if overlap else piece)
    return result


def _tail_tokens(text: str, n: int) -> str:
    tokens = _ENC.encode(text)
    return _ENC.decode(tokens[-n:]) if len(tokens) > n else text


def _align_to_sentence_start(text: str) -> str:
    """丢掉开头那半句，从句末符之后开始。找不到句末符则返回空串（不加重叠）。"""
    for i, ch in enumerate(text):
        if ch in "。！？；!?;\n":
            return text[i + 1 :].lstrip()
    return ""
