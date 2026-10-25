from pathlib import Path


def test_chat_page_posts_feedback_and_deep_links():
    html = Path("static/index.html").read_text(encoding="utf-8")
    assert "/api/feedback" in html
    assert "message_id" in html
    assert "URLSearchParams" in html
    assert "/review" in html and "/observability" in html
    assert "attachFeedback(bubbleEl)" not in html  # 只给带 message_id 的消息挂载
