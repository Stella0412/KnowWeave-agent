import pytest

from app.rag.evidence import (
    Evidence,
    assemble,
    build_evidence_block,
    evidence_from_hit,
    head_tail_order,
    to_citations,
)


def _ev(i, score=0.5, **over):
    payload = {
        "id": i, "question": f"问{i}", "answer": f"答{i}", "section_path": f"文档 > 节{i}",
        "content_type": "policy", "row_category": "", "category": "文档", "score": score,
    }
    payload.update(over)
    return Evidence(**payload)


def test_head_tail_order_matches_spec():
    """spec §7.4 给的就是这一串：最相关的落最前和最后，最不相关的沉到中间。"""
    assert head_tail_order(10) == [0, 2, 4, 6, 8, 9, 7, 5, 3, 1]


def test_head_tail_order_odd_count_keeps_second_best_last():
    assert head_tail_order(5) == [0, 2, 4, 3, 1]
    assert head_tail_order(1) == [0]
    assert head_tail_order(0) == []


def test_head_tail_order_puts_best_first_and_runner_up_last():
    order = head_tail_order(4)
    assert order[0] == 0
    assert order[-1] == 1


def test_assemble_renumbers_by_assembly_order():
    items = [_ev(i) for i in range(10)]
    out = assemble(items)
    assert [e.id for e in out] == [0, 2, 4, 6, 8, 9, 7, 5, 3, 1]
    # 编号按装配序赋：prompt 里模型先读到的那条必须是 [1]
    assert [e.n for e in out] == list(range(1, 11))


def test_assemble_does_not_mutate_input_scores():
    items = [_ev(i, score=1.0 - i / 10) for i in range(3)]
    out = assemble(items)
    assert [e.score for e in out] == [1.0, pytest.approx(1.0 - 2 / 10), pytest.approx(1.0 - 1 / 10)]


def test_evidence_from_hit_maps_all_fields():
    hit = {
        "id": 42, "score": -3.25, "question": "退款", "answer": "3 个工作日",
        "category": "售后手册", "content_type": "manual",
        "section_path": "售后手册 > 常见问题处理时限", "row_category": "退款",
    }
    ev = evidence_from_hit(hit)
    assert ev.id == 42 and ev.score == -3.25 and ev.row_category == "退款"
    assert ev.doc_root == "售后手册"
    assert ev.rerank_text == "退款\n3 个工作日"


def test_doc_root_of_missing_section_path_is_empty():
    assert _ev(1, section_path="").doc_root == ""


def test_build_evidence_block_prefixes_each_item_with_its_number():
    items = assemble([_ev(7), _ev(8)])
    block = build_evidence_block(items)
    assert block.startswith("[1]")
    assert "[2]" in block
    assert "问8" in block and "答7" in block


def test_build_evidence_block_marks_row_category():
    items = assemble([_ev(1, row_category="退款", section_path="售后手册 > 常见问题处理时限")])
    block = build_evidence_block(items)
    assert "售后手册 > 常见问题处理时限" in block
    assert "退款" in block


def test_to_citations_payload_shape_and_order():
    items = assemble([_ev(7, score=0.9), _ev(8, score=0.2)])
    payload = to_citations(items, source_files={"文档": "doc.md"})
    assert payload == [
        {
            "n": 1, "chunk_id": 7, "doc_root": "文档", "source_file": "doc.md",
            "section_path": "文档 > 节7", "question": "问7", "answer": "答7",
            "content_type": "policy", "row_category": None, "score": 0.9,
        },
        {
            "n": 2, "chunk_id": 8, "doc_root": "文档", "source_file": "doc.md",
            "section_path": "文档 > 节8", "question": "问8", "answer": "答8",
            "content_type": "policy", "row_category": None, "score": 0.2,
        },
    ]


def test_to_citations_without_source_files_gives_null_source_file():
    """原始文件名没落库、KB 目录也不可见时，宁可不给入口，也不硬编文件名（spec §12）。"""
    payload = to_citations(assemble([_ev(1)]), source_files={})
    assert payload[0]["source_file"] is None


def test_to_citations_marks_bm25_scores_as_uncomparable():
    """score 在三种策略下量纲不同，前端只做展示；负数（BM25）也照样返回。"""
    payload = to_citations(assemble([_ev(1, score=-3.25)]), source_files={})
    assert payload[0]["score"] == -3.25
