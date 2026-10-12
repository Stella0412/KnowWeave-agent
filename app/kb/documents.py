"""源文档 → chunk 记录的单一实现。CLI 与未来页面都调 build_chunks，避免将来分叉。"""

from app.kb.chunking import Chunk, ChunkLimits, split_markdown


def build_chunks(
    text: str, *, content_type: str, limits: ChunkLimits | None = None
) -> list[Chunk]:
    """把一篇 Markdown 正文切成 chunk 记录（字段映射在 chunking.split_markdown 内完成）。"""
    return split_markdown(text, default_content_type=content_type, limits=limits)
