"""裁判回归判例的结构检查：不调 API，先把「判例本身靠不靠谱」钉住。"""

import json
import re
from pathlib import Path

from app.prompts.rag import FAITH_JUDGE_SYSTEM

CASES_PATH = Path(__file__).resolve().parents[1] / "eval" / "judge_cases.jsonl"


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_fixture_has_both_labels_and_unique_ids():
    cases = load_cases()
    assert len(cases) == 18
    ids = [case["id"] for case in cases]
    assert len(set(ids)) == len(ids)
    faithful = sum(1 for case in cases if case["expected_faithful"])
    assert faithful == 11
    assert len(cases) - faithful == 7


def test_every_case_has_query_answer_evidence_and_note():
    for case in load_cases():
        for field in ("id", "query", "answer", "evidence", "expected_faithful", "note"):
            assert case.get(field) not in (None, ""), f"{case.get('id')} 缺 {field}"
        assert isinstance(case["expected_faithful"], bool)


def test_answer_citation_numbers_exist_in_evidence():
    for case in load_cases():
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", case["answer"])}
        available = {
            int(n) for n in re.findall(r"^\[(\d+)\]", case["evidence"], flags=re.MULTILINE)
        }
        assert cited <= available, f"{case['id']} 引用了不存在的编号：{sorted(cited - available)}"


def test_fixture_covers_case_study_and_real_cases():
    cases = load_cases()
    sources = {case["source"] for case in cases}
    assert sources == {"case_study", "latest_20261017"}
    assert sum(1 for case in cases if case["source"] == "case_study") == 6


def test_judge_prompt_has_seven_exemptions_and_focus_classes():
    for keyword in (
        "平台既定的服务渠道引导",
        "兜底表述",
        "合理拒答",
        "跨证据合并",
        "不确定语气的提醒",
        "无害的安全提示",
        "通用常识注解",
        "凭空给出的具体数值",
        "条件张冠李戴",
        "与资料明确写明的规则直接相反",
    ):
        assert keyword in FAITH_JUDGE_SYSTEM


def test_judge_prompt_has_six_boundary_examples_and_conclusion_first():
    boundary = FAITH_JUDGE_SYSTEM.split("## 边界样例", 1)[1]
    assert boundary.count("\n- ") == 6
    assert boundary.count("→ 忠实") == 4
    assert boundary.count("→ 编造") == 2
    assert "结论：忠实" in FAITH_JUDGE_SYSTEM
    assert "结论：编造" in FAITH_JUDGE_SYSTEM
