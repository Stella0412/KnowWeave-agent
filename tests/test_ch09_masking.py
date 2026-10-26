from types import SimpleNamespace

from app.observability.masking import mask_otel_spans, mask_text


def test_mask_text_covers_pii():
    assert mask_text("身份证 11010519491231002X") == "身份证 [REDACTED ID]"
    assert mask_text("电话 +86 13800138000") == "电话 [REDACTED PHONE]"
    assert mask_text("邮箱 zhi.xu@example.com") == "邮箱 [REDACTED EMAIL]"
    assert mask_text("卡号 6222 0212 3456 7890 123") == "卡号 [REDACTED CARD]"


def test_mask_text_keeps_decimal_scores():
    for value in ("0.9845678901234567", '{"top_score": 0.1234567890123456}'):
        assert mask_text(value) == value


def test_mask_otel_spans_patches_only_changed_attributes():
    params = SimpleNamespace(
        spans={
            "a": SimpleNamespace(
                attributes={
                    "gen_ai.input.messages": "联系 13800138000",
                    "app.score": 0.9845678901234567,
                }
            ),
            "b": SimpleNamespace(attributes={"app.note": "no pii"}),
        }
    )

    result = mask_otel_spans(params=params)

    assert set(result.span_patches) == {"a"}
    assert result.span_patches["a"].set_attributes == {
        "gen_ai.input.messages": "联系 [REDACTED PHONE]",
    }
