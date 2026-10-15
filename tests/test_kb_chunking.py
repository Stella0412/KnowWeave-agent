from app.kb.chunking import (
    ChunkLimits,
    count_tokens,
    first_heading,
    is_table_block,
    split_markdown,
    trim_to_sentence_end,
)

LIMITS = ChunkLimits()


def test_first_heading():
    assert first_heading("# 退货与运费政策\n## 甲\n正文") == "退货与运费政策"
    assert first_heading("没有标题\n正文") is None


def test_policy_maps_questions_to_section_title_and_category_to_parent_path():
    text = """# 退货与运费政策

## 运费与邮费说明

单笔订单满 99 元包邮，不满 99 元收 8 元运费。
"""
    chunks = split_markdown(text, default_content_type="policy", limits=LIMITS)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.questions == "运费与邮费说明"
    assert c.category == "退货与运费政策"
    assert c.section_path == "退货与运费政策 > 运费与邮费说明"
    assert c.content_type == "policy"
    assert "满 99 元包邮" in c.answer


def test_manual_deep_path():
    text = """# 售后手册

## 退款流程

### 审核

客服在 24 小时内审核退款申请。
"""
    chunks = split_markdown(text, default_content_type="manual", limits=LIMITS)
    assert [c.section_path for c in chunks] == ["售后手册 > 退款流程 > 审核"]
    assert chunks[0].questions == "审核"
    assert chunks[0].category == "售后手册 > 退款流程"


def test_faq_uses_real_question_and_alias_line():
    text = """# 商品常见问题

## 这款猫粮多少钱

问法：猫粮价格 / 猫粮贵不贵

售价 89 元一袋，规格 2kg。
"""
    chunks = split_markdown(text, default_content_type="faq", limits=LIMITS)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.questions == "这款猫粮多少钱\n猫粮价格\n猫粮贵不贵"
    assert "问法：" not in c.answer
    assert c.category == "商品常见问题"
    assert c.content_type == "faq"


def test_doc_without_heading_yields_no_chunks():
    assert split_markdown("只有正文，没有标题", default_content_type="md", limits=LIMITS) == []


def test_empty_body_sections_are_skipped():
    text = "# 标题\n\n## 空章节\n\n## 有内容\n\n正文内容。\n"
    chunks = split_markdown(text, default_content_type="policy", limits=LIMITS)
    assert [c.questions for c in chunks] == ["有内容"]


def test_key_clause_marker_sets_flag():
    text = """# 退货与运费政策

## 生鲜退换

> 【关键条款】生鲜类商品一经签收不支持 7 天无理由退货。

其余商品正常适用。
"""
    chunks = split_markdown(text, default_content_type="policy", limits=LIMITS)
    assert chunks[0].is_key_clause is True
    assert "【关键条款】" in chunks[0].answer


def test_plain_clause_is_not_flagged():
    text = "# 政策\n\n## 普通条款\n\n普通说明文字。\n"
    assert split_markdown(text, default_content_type="policy", limits=LIMITS)[0].is_key_clause is False


def test_long_section_is_split_into_multiple_chunks():
    sentence = "这是用来撑长度的句子，重复出现以保证超过块上限。"  # 约 24 token
    long_body = sentence * 80
    limits = ChunkLimits(target_tokens=300, max_tokens=400, overlap_tokens=40)
    chunks = split_markdown(
        "# 题目\n\n## 超长章节\n\n" + long_body, default_content_type="policy", limits=limits
    )
    assert len(chunks) > 3
    assert all(count_tokens(c.answer) <= limits.max_tokens for c in chunks)


def test_chunks_end_at_sentence_boundary():
    sentence = "第一句说的是甲。第二句说的是乙。第三句说的是丙。"
    limits = ChunkLimits(target_tokens=12, max_tokens=20, overlap_tokens=0)
    chunks = split_markdown(
        "# 题目\n\n## 章节\n\n" + sentence * 6, default_content_type="policy", limits=limits
    )
    for c in chunks:
        assert c.answer.rstrip()[-1] in "。！？；"


def test_overlap_text_is_appended_to_next_chunk():
    sentence = "甲句内容。乙句内容。丙句内容。"
    limits = ChunkLimits(target_tokens=12, max_tokens=20, overlap_tokens=8)
    chunks = split_markdown(
        "# 题目\n\n## 章节\n\n" + sentence * 6, default_content_type="policy", limits=limits
    )
    assert len(chunks) >= 2
    tail_sentence = chunks[0].answer.rstrip().split("。")[-2] + "。"
    assert tail_sentence in chunks[1].answer


def test_single_oversized_sentence_is_hard_split():
    no_punct = "甲" * 2000
    limits = ChunkLimits(target_tokens=100, max_tokens=150, overlap_tokens=0)
    chunks = split_markdown(
        "# 题目\n\n## 章节\n\n" + no_punct, default_content_type="policy", limits=limits
    )
    assert len(chunks) > 1
    assert all(count_tokens(c.answer) <= limits.max_tokens for c in chunks)


def test_trim_to_sentence_end():
    assert trim_to_sentence_end("完整一句。半截话") == "完整一句。"
    assert trim_to_sentence_end("没有句末符") == "没有句末符"
    assert trim_to_sentence_end("完整一句。") == "完整一句。"


BIG_TABLE = """# 商品常见问题

## 各品类配送时效

| 序号 | 配送时效 | 是否包邮 |
| --- | --- | --- |
""" + "".join(f"| 序号{i} | {i}天 | 是 |\n" for i in range(60))


def test_big_table_is_split_by_rows_with_header_repeated():
    limits = ChunkLimits(target_tokens=200, max_tokens=400, overlap_tokens=0, table_max_rows=10)
    chunks = split_markdown(BIG_TABLE, default_content_type="faq", limits=limits)
    assert len(chunks) > 1
    for c in chunks:
        assert "| 序号 | 配送时效 | 是否包邮 |" in c.answer  # 每块都复制表头
        assert "| --- | --- | --- |" in c.answer  # 分隔行也复制
        assert c.row_category is None      # 非白名单表不记分类


def test_table_rows_are_not_cut_in_half():
    limits = ChunkLimits(target_tokens=200, max_tokens=400, overlap_tokens=0, table_max_rows=10)
    chunks = split_markdown(BIG_TABLE, default_content_type="faq", limits=limits)
    for c in chunks:
        for line in c.answer.splitlines():
            assert line.count("|") in (0, 4)  # 每行要么非表格，要么是完整的一行四列


def test_small_table_stays_intact():
    text = """# 商品常见问题

## 价格

| 商品 | 价格 |
| --- | --- |
| 猫粮 | 89 |
"""
    chunks = split_markdown(text, default_content_type="faq", limits=LIMITS)
    assert len(chunks) == 1
    assert "| 猫粮 | 89 |" in chunks[0].answer


def test_split_table_chunks_start_with_header_even_with_overlap():
    """表格片自带表头，不能被重叠顶掉（spec §6「每组重贴表头行 + 分隔行」）。

    `_apply_overlap` 曾对表格片也接上一片尾巴：表格行以 "\n" 结尾，
    `_align_to_sentence_start` 会从第一个换行之后取值 → 下一片被塞进一行**数据行**，
    表头不再是首行，且边界那行重复出现。
    """
    limits = ChunkLimits(target_tokens=200, max_tokens=400, overlap_tokens=80, table_max_rows=10)
    chunks = split_markdown(BIG_TABLE, default_content_type="faq", limits=limits)
    assert len(chunks) > 1
    for c in chunks:
        lines = c.answer.splitlines()
        assert lines[0].strip() == "| 序号 | 配送时效 | 是否包邮 |"
        assert lines[1].strip() == "| --- | --- | --- |"
