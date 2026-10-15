from app.kb.chunking import (
    CATEGORY_COLUMN_NAMES,
    ChunkLimits,
    find_category_column,
    split_markdown,
)

LIMITS = ChunkLimits()

# 与 data/kb/after-sales-manual.md > 常见问题处理时限 同形（12 行、只有分类列不同）
HOMOGENEOUS_TABLE = """# 售后手册

## 常见问题处理时限

| 问题类型 | 首次响应 | 处理时限 |
| --- | --- | --- |
| 退款 | 2 小时 | 3 个工作日 |
| 换货 | 2 小时 | 5 个工作日 |
| 维修 | 4 小时 | 7 个工作日 |
"""


def test_find_category_column_matches_whitelist_only():
    assert find_category_column("| 问题类型 | 首次响应 | 处理时限 |") == 0
    assert find_category_column("| 品类 | 快递 |") == 0
    assert find_category_column("| 型号 | 容量 |") == 0
    assert find_category_column("| 序号 | 说明 |") is None
    assert find_category_column("| 商品 | 价格 |") is None


def test_whitelist_is_defined_in_one_place():
    assert "问题类型" in CATEGORY_COLUMN_NAMES and "型号" in CATEGORY_COLUMN_NAMES


def test_category_table_is_split_per_row_even_when_short():
    """短表也按行拆：整表一块就横跨 12 个分类，给不出单一 row_category（spec §6）。"""
    chunks = split_markdown(HOMOGENEOUS_TABLE, default_content_type="manual", limits=LIMITS)
    assert [c.row_category for c in chunks] == ["退款", "换货", "维修"]
    for c in chunks:
        assert c.section_path == "售后手册 > 常见问题处理时限"
        # 每块自带表头 + 分隔行，单看一块也读得懂
        assert c.answer.splitlines()[0].strip() == "| 问题类型 | 首次响应 | 处理时限 |"
        assert c.answer.splitlines()[1].strip() == "| --- | --- | --- |"


def test_each_row_chunk_contains_only_its_own_row():
    chunks = split_markdown(HOMOGENEOUS_TABLE, default_content_type="manual", limits=LIMITS)
    assert "| 退款 | 2 小时 | 3 个工作日 |" in chunks[0].answer
    assert "换货" not in chunks[0].answer


def test_normal_table_without_category_column_keeps_ch03_behavior():
    text = """# 商品常见问题

## 价格

| 商品 | 价格 |
| --- | --- |
| 猫粮 | 89 |
"""
    chunks = split_markdown(text, default_content_type="faq", limits=LIMITS)
    assert len(chunks) == 1
    assert chunks[0].row_category is None


def test_plain_text_chunk_has_no_row_category():
    text = "# 售后手册\n\n## 保修说明\n\n保修 12 个月。\n"
    chunks = split_markdown(text, default_content_type="manual", limits=LIMITS)
    assert chunks[0].row_category is None


def test_ragged_row_does_not_crash_and_falls_back():
    """人工编辑 KB 时少写一列 → 下标越界不能抛 IndexError，退化成普通表处理（Review Focus 5）。"""
    text = """# 售后手册

## 常见问题处理时限

| 问题类型 | 首次响应 | 处理时限 |
| --- | --- | --- |
| 退款 | 2 小时 |
"""
    chunks = split_markdown(text, default_content_type="manual", limits=LIMITS)
    assert len(chunks) == 1
    assert chunks[0].row_category is None   # 缺分类格 → 退化成普通表，不猜
    assert "退款" in chunks[0].answer


def test_row_category_is_truncated_to_column_width():
    long_name = "分类" * 40
    text = (
        "# 售后手册\n\n## 时限\n\n| 问题类型 | 处理时限 |\n| --- | --- |\n"
        f"| {long_name} | 3 个工作日 |\n"
    )
    chunks = split_markdown(text, default_content_type="manual", limits=LIMITS)
    assert chunks[0].row_category is not None
    assert len(chunks[0].row_category) <= 64
