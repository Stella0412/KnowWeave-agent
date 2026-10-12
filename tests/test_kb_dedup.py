from app.kb.dedup import normalize_text, qa_fingerprint


def test_normalize_text_ignores_space_case_and_punctuation():
    assert normalize_text(" 邮费是多少？ ") == normalize_text("邮费是多少")
    assert normalize_text("ABC １２３") == normalize_text("abc123")
    assert normalize_text("问：能开发票吗！") == normalize_text("问能开发票吗")


def test_qa_fingerprint_uses_question_and_answer():
    a = qa_fingerprint("能开发票吗", "可以")
    b = qa_fingerprint("能开发票吗！", "可以。")
    assert a == b  # 归一化后同指纹
    assert qa_fingerprint("能开发票吗", "可以") != qa_fingerprint("能开发票吗", "不可以")
