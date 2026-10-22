"""退款申请资格：只允许仍在无理由退货期内、且不属于除外类目的已签收订单填表。

口径来自 ``data/kb/returns-policy.md``：自签收之日起 7 天内、商品完好的普通商品
支持无理由退货；定制类、生鲜类商品除外；已取消/未签收的订单没有可退的在途交易。
"""

from datetime import datetime, timedelta

REFUND_WINDOW_DAYS = 7
EXCLUDED_PRODUCT_TYPES = frozenset({"定制", "生鲜"})


def _field(order, name: str):
    if isinstance(order, dict):
        return order.get(name)
    return getattr(order, name, None)


def _as_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def check_refund_eligibility(
    order, *, now: datetime | None = None
) -> tuple[bool, str]:
    """返回 ``(是否可申请退款, 不可申请的原因)``。"""
    status = str(_field(order, "status") or "").strip()
    product_type = str(_field(order, "product_type") or "").strip()
    signed_at = _as_datetime(_field(order, "signed_at"))
    current = now or datetime.now()
    # DB 里是无时区的本地时间；若调用方传入带时区的值，按本地墙钟比较。
    if signed_at is not None and signed_at.tzinfo is not None:
        signed_at = signed_at.replace(tzinfo=None)
    if current.tzinfo is not None:
        current = current.replace(tzinfo=None)

    if status == "已取消":
        return False, "订单已取消，不存在需要退款的在途交易"
    if status != "已签收" or signed_at is None:
        return False, "订单尚未签收，暂不能按无理由退货申请退款"
    if product_type in EXCLUDED_PRODUCT_TYPES:
        return False, f"{product_type}类商品不支持无理由退货"
    if signed_at + timedelta(days=REFUND_WINDOW_DAYS) < current:
        return False, f"已超过签收后 {REFUND_WINDOW_DAYS} 天的无理由退货期限"
    return True, ""
