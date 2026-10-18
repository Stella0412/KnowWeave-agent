"""RAG 评估面板的纯计算部分：证据覆盖、答案覆盖、排序名次。

这些函数不碰 Milvus / LLM，先把「页面上的数字怎么算」钉死，避免面板只是把报告抄一遍。
"""

import json

from app.config import settings
from app.services import rag_eval_service
from app.services.rag_eval_service import (
    _answer_summary,
    answer_coverage,
    evidence_coverage,
    expected_tokens,
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


def test_token_coverage_ignores_markdown_spacing_and_citations():
    cov = token_coverage(
        ["5000 元起", "原路退回 [1]"],
        "累计消费 **5000 元** 起升金卡；审核通过后原路退回 [10]。",
    )
    assert cov["missing"] == []
    assert cov["rate"] == 1.0


def test_token_coverage_accepts_minor_wording_insertions():
    cov = token_coverage(
        ["30 天内可申请开票", "快递费到付"],
        "订单完成后 30 天内可以申请开票；纸质发票快递费为到付。",
    )
    assert cov["missing"] == []
    assert cov["rate"] == 1.0


def test_token_coverage_accepts_reordered_short_phrase():
    cov = token_coverage(["1 次/月"], "银卡会员每月享有 1 次退换货运费权益。")
    assert cov["missing"] == []
    assert cov["rate"] == 1.0


def test_token_coverage_still_rejects_missing_numbers_and_facts():
    cov = token_coverage(["6 元运费", "平台承担"], "运费 10 元，由买家承担。")
    assert cov["matched"] == []
    assert cov["rate"] == 0.0


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


def test_expected_tokens_prefers_expected_points_over_must_contain():
    sample = {
        "expected_points": ["满 99 元包邮", "6 元基础运费"],
        "ground_truth": {"must_contain": ["6 元"]},
    }
    assert expected_tokens(sample) == ["满 99 元包邮", "6 元基础运费"]


def test_expected_tokens_falls_back_to_must_contain():
    sample = {"ground_truth": {"must_contain": ["6 元"]}}
    assert expected_tokens(sample) == ["6 元"]


def test_answer_coverage_uses_expected_points_not_only_must_contain():
    # must_contain 只负责收窄检索相关块；答案覆盖要看全部标准要点。
    sample = {
        "expected_behavior": "answer",
        "expected_points": ["满 99 元包邮", "6 元基础运费"],
        "ground_truth": {"must_contain": ["6 元"]},
    }
    cov = answer_coverage(sample, "满 99 元包邮。", refused=False)
    assert cov["matched"] == ["满 99 元包邮"]
    assert cov["missing"] == ["6 元基础运费"]
    assert cov["ok"] is False


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


def test_answer_coverage_for_no_commitment_also_requires_expected_points():
    # 不承诺到账时间只是底线；题目还问了「退到哪里」，不能漏。
    sample = {
        "expected_behavior": "no_commitment",
        "expected_points": ["原路退回"],
        "ground_truth": {},
    }
    cov = answer_coverage(sample, "平台不承诺具体到账时间，以支付渠道实际到账为准。", refused=False)
    assert cov["ok"] is False
    assert cov["missing"] == ["原路退回"]


def test_answer_coverage_for_no_commitment_fails_when_refused():
    # 该答的题被拒答，不能因为「没出现到账承诺」就算覆盖
    sample = {"expected_behavior": "no_commitment", "ground_truth": {"must_contain": []}}
    cov = answer_coverage(sample, "抱歉，暂时没有查到相关信息。", refused=True)
    assert cov["ok"] is False


def test_answer_summary_counts_answerable_questions_refused_by_mistake():
    # D_absent 的正确拒答不算误拒；可回答题被拒才要单独报出来，
    # 否则答案覆盖率掉下来时看不出是检索/自检把该答的题拦了。
    questions = [
        {"bucket": "A_policy", "refused": True, "answer_coverage": {"ok": False}},
        {"bucket": "A_policy", "refused": False, "answer_coverage": {"ok": True}},
        {"bucket": "D_absent", "refused": True, "answer_coverage": {"ok": True}},
    ]
    summary = _answer_summary(questions)
    assert summary["refused_answerable"] == 1
    assert summary["answerable_total"] == 2
    assert summary["refusal_correct"] == 1
    assert summary["refusal_total"] == 1


def test_read_latest_attaches_models_from_settings(tmp_path, monkeypatch):
    # 旧结果文件里没有模型名；页头的「嵌入/重排」「裁判模型」按当前配置补，
    # 免得为了看模型名重跑一轮 300 题。
    path = tmp_path / "latest.json"
    path.write_text(json.dumps({"run_id": "r1", "samples": 3}), encoding="utf-8")
    monkeypatch.setattr(rag_eval_service, "LATEST_PATH", path)

    data = rag_eval_service.read_latest()

    assert data["run_id"] == "r1"
    assert data["models"] == {
        "embed": settings.embed_model,
        "rerank": settings.rerank_model,
        "judge": settings.llm_model,
    }


def test_read_latest_keeps_existing_models_snapshot(tmp_path, monkeypatch):
    # 新结果已经带模型快照时不要被当前配置覆盖：报告描述的是那一轮用的模型。
    path = tmp_path / "latest.json"
    saved = {"embed": "old-embed", "rerank": "old-rerank", "judge": "old-judge"}
    path.write_text(json.dumps({"run_id": "r1", "models": saved}), encoding="utf-8")
    monkeypatch.setattr(rag_eval_service, "LATEST_PATH", path)

    assert rag_eval_service.read_latest()["models"] == saved
