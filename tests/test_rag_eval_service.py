"""RAG 评估面板的纯计算部分：证据覆盖、答案覆盖、排序名次。

这些函数不碰 Milvus / LLM，先把「页面上的数字怎么算」钉死，避免面板只是把报告抄一遍。
"""

from app.services.rag_eval_service import (
    answer_coverage,
    evidence_coverage,
    first_relevant_rank,
    token_coverage,
)


def test_token_coverage_counts_each_expected_fact_once():
    cov = token_coverage(["6 元", "包邮", "不存在"], "满 99 元包邮，运费 6 元")
    assert cov["matched"] == ["6 元", "包邮"]
    assert cov["missing"] == ["不存在"]
    assert cov["rate"] == 2 / 3


def test_token_coverage_without_tokens_is_not_a_score():
    cov = token_coverage([], "任意文本")
    assert cov["rate"] is None
    assert cov["matched"] == []
    assert cov["missing"] == []


def test_evidence_coverage_uses_must_contain_tokens():
    hits = [
        {"id": 1, "question": "运费怎么算", "answer": "满 99 元包邮，运费 6 元"},
        {"id": 2, "question": "发票", "answer": "电子普通发票"},
    ]
    cov = evidence_coverage(hits, ["6 元", "电子普通发票"], {1})
    assert cov["rate"] == 1.0
    assert cov["basis"] == "must_contain"


def test_evidence_coverage_falls_back_to_relevant_chunk_when_no_tokens():
    hits = [{"id": 7, "question": "q", "answer": "a"}]
    cov = evidence_coverage(hits, [], {7})
    assert cov["rate"] == 1.0
    assert cov["basis"] == "relevant_chunk"


def test_first_relevant_rank_is_one_based_and_none_when_missing():
    assert first_relevant_rank([9, 1, 2], {2}) == 3
    assert first_relevant_rank([9, 8], {2}) is None


def test_answer_coverage_for_answer_behavior_requires_all_key_facts():
    sample = {
        "expected_behavior": "answer",
        "ground_truth": {"must_contain": ["6 元"]},
    }
    good = answer_coverage(sample, "普通地区运费 6 元。", refused=False)
    bad = answer_coverage(sample, "普通地区运费 8 元。", refused=False)
    assert good["ok"] is True and good["rate"] == 1.0
    assert bad["ok"] is False and bad["rate"] == 0.0


def test_answer_coverage_for_refuse_behavior_checks_refusal():
    sample = {"expected_behavior": "refuse", "ground_truth": {}}
    assert answer_coverage(sample, "", refused=True)["ok"] is True
    assert answer_coverage(sample, "我不知道", refused=False)["ok"] is False


def test_answer_coverage_for_no_commitment_checks_negative_assertion():
    sample = {
        "expected_behavior": "no_commitment",
        "ground_truth": {"must_contain": ["不承诺"]},
    }
    good = answer_coverage(sample, "平台不承诺具体到账时间，以支付渠道实际到账为准。", refused=False)
    bad = answer_coverage(sample, "退款 3 个工作日到账。", refused=False)
    assert good["ok"] is True
    assert bad["ok"] is False
    assert bad["violations"]


def test_answer_coverage_for_no_commitment_fails_when_refused():
    # 该答的题被拒答，不能因为「没出现到账承诺」就算覆盖
    sample = {"expected_behavior": "no_commitment", "ground_truth": {"must_contain": []}}
    cov = answer_coverage(sample, "抱歉，暂时没有查到相关信息。", refused=True)
    assert cov["ok"] is False
