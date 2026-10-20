"""自评闸门的口径：证据「部分支持」也要答，只有「完全没写」才拒。

真正的行为验证是 300 题 full 评估；这里钉住 prompt 里那几条不能被删掉的判据。
"""

from app.prompts.rag import SELF_CHECK_SYSTEM


def test_self_check_prompt_allows_partial_answers():
    assert "只支持其中一部分" in SELF_CHECK_SYSTEM
    assert "不要整题拒答" in SELF_CHECK_SYSTEM


def test_self_check_prompt_still_rejects_topical_near_misses():
    assert "沾边" in SELF_CHECK_SYSTEM
    assert "完全没写" in SELF_CHECK_SYSTEM
