import pytest

from ml.taxonomy import TaxonomyError, load_taxonomy

EXPECTED = (
    "退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动", "价保",
    "支付", "订单修改", "库存补货", "商品信息", "保修维修", "账号", "会员积分",
    "评价", "其他",
)


def test_taxonomy_has_exact_17_labels_in_order():
    tax = load_taxonomy()
    assert tax.names == EXPECTED
    assert len(tax.names) == 17
    assert tax.names[-1] == "其他"


def test_every_category_has_definition_and_boundary():
    tax = load_taxonomy()
    for category in tax.categories:
        assert category.definition.strip()
        assert category.boundary.strip()
        assert category.examples, category.name
        assert all(name in tax.names for name in category.confusable_with)


def test_boundary_phrases_are_written_into_table():
    tax = load_taxonomy()
    text = tax.prompt_table()
    for phrase in ("修归保修维修", "退归退换货", "运费管钱", "物流管货", "价保是补差价", "优惠活动是券和满减"):
        assert phrase in text


def test_multi_hot_and_ordered_respect_taxonomy_order():
    tax = load_taxonomy()
    vec = tax.multi_hot(["退换货", "尺码"])
    assert len(vec) == 17
    assert vec[tax.index("尺码")] == 1.0
    assert vec[tax.index("退换货")] == 1.0
    assert sum(vec) == 2.0
    assert tax.ordered({"退换货", "尺码", "退换货"}) == ["退换货", "尺码"]


def test_unknown_label_raises():
    tax = load_taxonomy()
    with pytest.raises(TaxonomyError):
        tax.multi_hot(["不存在的类"])


def test_missing_file_raises(tmp_path):
    with pytest.raises(TaxonomyError):
        load_taxonomy(tmp_path / "nope.json")
