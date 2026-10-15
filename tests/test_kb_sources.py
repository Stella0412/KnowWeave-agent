from pathlib import Path

from app.kb.sources import detect_content_type, list_sources, source_files_by_root

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"

# 用户提供的 6 篇夹具，逐篇断言，防再退化成 md（spec §6.1）
EXPECTED = {
    "after-sales-manual.md": "manual",
    "billing-shipping.md": "policy",
    "member-benefits.md": "policy",
    "product-faq.md": "faq",
    "product-specs.md": "spec",
    "returns-policy.md": "policy",
}


def test_each_fixture_file_maps_to_expected_content_type():
    for filename, expected in EXPECTED.items():
        assert detect_content_type(filename) == expected, filename


def test_unknown_filename_falls_back_to_md():
    assert detect_content_type("随便什么.md") == "md"


def test_list_sources_returns_all_six_fixtures():
    sources = list_sources(KB_DIR)
    assert {s.path.name: s.content_type for s in sources} == EXPECTED


def test_source_files_by_root_maps_h1_to_filename():
    mapping = source_files_by_root(KB_DIR)
    assert mapping["售后手册"] == "after-sales-manual.md"
    assert mapping["商品规格手册"] == "product-specs.md"
    assert mapping["退货退款政策"] == "returns-policy.md"


def test_source_files_by_root_missing_dir_is_empty():
    assert source_files_by_root("data/不存在的目录") == {}
