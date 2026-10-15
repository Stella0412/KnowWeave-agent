from app.kb.chunking import Chunk, ChunkLimits
from app.kb.documents import build_chunks
from app.kb.sources import Source, detect_content_type, list_sources

LIMITS = ChunkLimits()


def test_detect_content_type_by_filename():
    assert detect_content_type("returns-policy.md") == "policy"
    assert detect_content_type("after-sales-manual.md") == "manual"
    assert detect_content_type("product-faq.md") == "faq"
    assert detect_content_type("unknown.md") == "md"


def test_list_sources_scans_markdown_only(tmp_path):
    (tmp_path / "b.md").write_text("# b", encoding="utf-8")
    (tmp_path / "a.md").write_text("# a", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("# h", encoding="utf-8")
    sources = list_sources(tmp_path)
    assert [s.path.name for s in sources] == ["a.md", "b.md"]
    assert all(isinstance(s, Source) for s in sources)


def test_build_chunks_is_single_implementation():
    """build_chunks 是 CLI 与未来页面的唯一入口；此处验证字段映射端到端正确。"""
    text = """# 退货与运费政策

## 运费与邮费说明

单笔订单满 99 元包邮。
"""
    chunks = build_chunks(text, content_type="policy", limits=LIMITS)
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].content_type == "policy"
    assert chunks[0].section_path == "退货与运费政策 > 运费与邮费说明"


def test_build_chunks_without_heading_yields_empty():
    assert build_chunks("只有正文没有标题", content_type="policy", limits=LIMITS) == []


def test_repo_source_docs_exist_and_contain_acceptance_answers():
    from pathlib import Path

    docs_dir = Path("data/kb")
    assert docs_dir.is_dir(), "data/kb 源文档目录必须存在"
    # 验收① 的答案必须在文档里（问「邮费是多少」要能召回）
    faq_text = (docs_dir / "product-faq.md").read_text(encoding="utf-8")
    assert "邮费" in faq_text and "运费" in faq_text and "99 元" in faq_text
