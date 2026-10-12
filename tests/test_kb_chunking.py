from app.kb.chunking import ChunkLimits, first_heading, split_markdown

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
