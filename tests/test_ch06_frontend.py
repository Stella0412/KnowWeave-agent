# tests/test_ch06_frontend.py
from pathlib import Path


def test_frontend_has_order_cards_and_refund_form():
    html = Path("static/index.html").read_text(encoding="utf-8")
    for marker in [
        "select_order",
        "renderOrderCards",
        "resumeWithOrder",
        "fetch('/chat/resume'",
        "refund_form",
        "openRefundModal",
        "fetch('/chat/refund'",
        "consumeStream",
        "refundReason",
    ]:
        assert marker in html, marker


def test_refund_modal_sets_order_number_without_html_interpolation():
    html = Path("static/index.html").read_text(encoding="utf-8")
    assert 'value="${draft.order_no' not in html
    assert "querySelector('#refundOrder').value = draft.order_no" in html
