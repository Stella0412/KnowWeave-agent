from datetime import datetime, timedelta
from types import SimpleNamespace

from app.services.refund_policy import check_refund_eligibility

NOW = datetime(2026, 10, 22, 12, 0, 0)


def _order(**overrides):
    data = {
        "order_no": "1001",
        "status": "已签收",
        "product_type": "普通",
        "signed_at": (NOW - timedelta(days=3)).isoformat(),
    }
    data.update(overrides)
    return data


def test_ordinary_order_within_seven_days_is_eligible():
    assert check_refund_eligibility(_order(), now=NOW) == (True, "")


def test_order_signed_exactly_seven_days_ago_is_still_eligible():
    signed = (NOW - timedelta(days=7)).isoformat()
    assert check_refund_eligibility(_order(signed_at=signed), now=NOW)[0] is True


def test_order_past_seven_days_is_not_eligible():
    signed = (NOW - timedelta(days=7, seconds=1)).isoformat()
    ok, reason = check_refund_eligibility(_order(signed_at=signed), now=NOW)
    assert ok is False
    assert "7 天" in reason


def test_custom_product_is_not_eligible():
    ok, reason = check_refund_eligibility(
        _order(product_type="定制"), now=NOW
    )
    assert ok is False
    assert "定制" in reason


def test_fresh_product_is_not_eligible():
    ok, reason = check_refund_eligibility(
        _order(product_type="生鲜"), now=NOW
    )
    assert ok is False
    assert "生鲜" in reason


def test_cancelled_order_is_not_eligible():
    ok, reason = check_refund_eligibility(
        _order(status="已取消", signed_at=None), now=NOW
    )
    assert ok is False
    assert "已取消" in reason


def test_unsigned_order_is_not_eligible():
    ok, reason = check_refund_eligibility(
        _order(status="已发货", signed_at=None), now=NOW
    )
    assert ok is False
    assert "尚未签收" in reason


def test_accepts_orm_objects_with_datetime_fields():
    order = SimpleNamespace(
        status="已签收",
        product_type="普通",
        signed_at=NOW - timedelta(days=2),
    )
    assert check_refund_eligibility(order, now=NOW) == (True, "")
