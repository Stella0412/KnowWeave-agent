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

# 「分类维度列」白名单：命中即把整张表按行拆、每行一块并记 row_category。
# 用白名单而不是「取第一列」——要的是语义确定的维度列，不是碰巧排在最前的那一列（spec §6）。
CATEGORY_COLUMN_NAMES = ("品类", "问题类型", "类型", "类别", "型号")
ROW_CATEGORY_MAX_LENGTH = 64

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
    # 分类维度表按行拆时该行分类列的取值；其余块留 None（不从自由正文猜分类）
    row_category: str | None = None


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
        for piece, row_category in _split_body(body, limits):
            chunks.append(
                Chunk(
                    category=category[:255],
                    questions=questions[:65535],
                    answer=piece,
                    section_path=section_path[:512],
                    content_type=default_content_type,
                    is_key_clause=KEY_CLAUSE_MARK in piece,
                    row_category=row_category,
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


def _split_body(body: str, limits: ChunkLimits) -> list[tuple[str, str | None]]:
    """正文 → [(块文本, row_category)]。

    两条路：① 含分类维度表的正文走块级拆分（表按行拆，其余块原样或递归切）；
    ② 其余完全沿用 ch03 行为（短正文整块返回），避免无谓的块数漂移。
    """
    lines = body.splitlines()
    blocks = split_blocks(lines)
    has_category_table = any(
        kind == "table" and find_category_column(block[0]) is not None for kind, block in blocks
    )
    if not has_category_table and count_tokens(body) <= limits.max_tokens:
        return [(body, None)]

    pieces: list[tuple[str, str, str | None]] = []
    for kind, block_lines in blocks:
        if kind == "table":
            column = find_category_column(block_lines[0])
            category_rows = (
                _split_category_table(block_lines, column) if column is not None else []
            )
            if category_rows:
                pieces.extend(("catrow", text, row_category) for text, row_category in category_rows)
            else:
                pieces.extend(("table", piece, None) for piece in _split_table(block_lines, limits))
        else:
            text = "\n".join(block_lines)
            if count_tokens(text) <= limits.max_tokens:
                pieces.append(("text", text, None))
            else:
                pieces.extend(
                    ("text", piece, None) for piece in _long_splitter(limits).split_text(text)
                )
    return _apply_overlap(pieces, limits.overlap_tokens)


def split_table_row(line: str) -> list[str]:
    """`| a | b |` → ["a", "b"]。表格行只有这一种解析方式，集中一处免得各写一遍。"""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def find_category_column(header_line: str) -> int | None:
    """表头里分类维度列的下标；没有则 None。"""
    for index, cell in enumerate(split_table_row(header_line)):
        if cell in CATEGORY_COLUMN_NAMES:
            return index
    return None


def _split_category_table(lines: list[str], column: int) -> list[tuple[str, str]]:
    """分类维度表按行拆：每块 = 表头 + 分隔行 + 1 行数据，返回 [(块文本, row_category)]。

    整行缺失分类格（列数不足）时整张表退化成「不拆」，由调用方兜底 —— 与其猜一个类别，
    不如老实当普通表（Review Focus 5）。
    """
    header, separator = lines[0].strip(), lines[1].strip()
    header_cells = split_table_row(header)
    pieces: list[tuple[str, str]] = []
    for row in lines[2:]:
        cells = split_table_row(row)
        # 行缺格（列数与表头不一致）或分类格缺失/为空 → 整表退化成普通表，不猜（Review Focus 5）。
        if len(cells) != len(header_cells) or column >= len(cells) or not cells[column]:
            return []
        pieces.append((f"{header}\n{separator}\n{row.strip()}\n", cells[column][:ROW_CATEGORY_MAX_LENGTH]))
    return pieces


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


def _apply_overlap(
    pieces: list[tuple[str, str, str | None]], overlap_tokens: int
) -> list[tuple[str, str | None]]:
    """给第 2 块起的每块头部接上前一块尾部 overlap_tokens（**仅 text→text 接缝**），并把接缝对齐到句首。

    表格块与分类行块完全不参与重叠（spec §6：表格片自带表头 + 分隔行，是自解释的）。理由有两层：
    1. 接缝排在**块首**：给表格片接上文本尾巴，表头就被挤到第 3 行，「自带表头」失效；
    2. 表格行的句末符是换行，`_align_to_sentence_start` 于是几乎总能切出一段尾巴 → 上一片的
       最后一行被复制到下一片开头，同一行在检索结果里出现两次。
    """
    if overlap_tokens <= 0 or len(pieces) < 2:
        return [(text, row_category) for _, text, row_category in pieces]
    result = [(pieces[0][1], pieces[0][2])]
    for (prev_kind, prev_text, _), (kind, text, row_category) in zip(pieces, pieces[1:]):
        if prev_kind != "text" or kind != "text":
            result.append((text, row_category))
            continue
        overlap = _align_to_sentence_start(_tail_tokens(prev_text, overlap_tokens))
        result.append(((overlap + text) if overlap else text, row_category))
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
