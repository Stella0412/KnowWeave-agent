import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_retrieval_ch04", Path(__file__).resolve().parent.parent / "eval" / "eval_retrieval_ch04.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_section_paths_of_supports_single_and_list():
    assert mod.section_paths_of({"ground_truth": {"section_path": "A > b"}}) == ["A > b"]
    assert mod.section_paths_of({"ground_truth": {"section_paths": ["A", "B"]}}) == ["A", "B"]
    assert mod.section_paths_of({"ground_truth": {}}) == []


def test_resolve_relevant_narrows_by_must_contain():
    rows = [
        {"id": 1, "section_path": "售后手册 > 常见问题处理时限", "question": "退款",
         "answer": "| 退款 | 2 小时 | 3 个工作日 |"},
        {"id": 2, "section_path": "售后手册 > 常见问题处理时限", "question": "换货",
         "answer": "| 换货 | 2 小时 | 5 个工作日 |"},
    ]
    sample = {"ground_truth": {"section_path": "售后手册 > 常见问题处理时限", "must_contain": ["退款"]}}
    assert mod.resolve_relevant(rows, sample) == {1}


def test_resolve_relevant_without_must_contain_takes_whole_section():
    rows = [{"id": 1, "section_path": "甲 > 乙", "question": "q", "answer": "a"}]
    assert mod.resolve_relevant(rows, {"ground_truth": {"section_path": "甲 > 乙"}}) == {1}


def test_resolve_relevant_narrows_one_section_without_touching_others():
    # 多章节题：12 行表按行收窄，其余章节整段保留
    rows = [
        {"id": 1, "section_path": "售后手册 > 常见问题处理时限", "question": "时限",
         "answer": "| 退款 | 2 小时 | 3 个工作日 |"},
        {"id": 2, "section_path": "售后手册 > 常见问题处理时限", "question": "时限",
         "answer": "| 换货 | 2 小时 | 5 个工作日 |"},
        {"id": 3, "section_path": "售后手册 > 人工客服入口与时间", "question": "客服",
         "answer": "人工客服服务时间为每日 9:00 至 22:00"},
    ]
    sample = {
        "ground_truth": {
            "section_paths": [
                "售后手册 > 常见问题处理时限",
                "售后手册 > 人工客服入口与时间",
            ],
            "must_contain_any_by_section": {
                "售后手册 > 常见问题处理时限": ["退款", "换货"],
            },
        }
    }
    assert mod.resolve_relevant(rows, sample) == {1, 2, 3}


def test_resolve_relevant_any_tokens_keep_only_matching_rows():
    rows = [
        {"id": 1, "section_path": "售后手册 > 常见问题处理时限", "question": "时限",
         "answer": "| 退款 | 2 小时 | 3 个工作日 |"},
        {"id": 2, "section_path": "售后手册 > 常见问题处理时限", "question": "时限",
         "answer": "| 换货 | 2 小时 | 5 个工作日 |"},
        {"id": 3, "section_path": "售后手册 > 常见问题处理时限", "question": "时限",
         "answer": "| 投诉 | 1 小时 | 3 个工作日 |"},
    ]
    sample = {
        "ground_truth": {
            "section_path": "售后手册 > 常见问题处理时限",
            "must_contain_any_by_section": {
                "售后手册 > 常见问题处理时限": ["退款", "投诉"],
            },
        }
    }
    assert mod.resolve_relevant(rows, sample) == {1, 3}


def test_resolve_relevant_empty_ground_truth_for_reject_bucket():
    assert mod.resolve_relevant([{"id": 1, "section_path": "甲", "question": "", "answer": ""}], {"ground_truth": {}}) == set()


def test_score_one_computes_recall_and_mrr():
    metrics = mod.score_one([9, 1, 2, 3, 4, 5], {1, 2}, k_values=(5, 10))
    assert metrics["recall@5"] == 1.0        # 1 与 2 都在前 5
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@10"] == 1 / 2        # 1 排在第 2 位


def test_score_one_partial_recall():
    metrics = mod.score_one([1, 8, 9], {1, 2}, k_values=(2,))
    assert metrics["recall@2"] == 0.5
    assert metrics["mrr@10"] == 1.0


def test_score_one_reject_bucket_returns_empty():
    assert mod.score_one([1, 2], set()) == {}
