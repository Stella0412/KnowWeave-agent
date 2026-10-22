# tests/test_ch07_frontend.py
from pathlib import Path


def test_sidebar_and_history_functions_exist():
    html = Path("static/index.html").read_text(encoding="utf-8")
    for marker in [
        'id="conversationList"',
        "loadConversations",
        "renderConversationList",
        "openConversation",
        "fetch('/api/conversations')",
        "fetch('/api/conversations/' + id + '/messages')",
        "conversation-item",
        "已摘要",
        "会话列表暂不可用",
        "btnSidebar",
        ".sidebar.open",
    ]:
        assert marker in html, marker


def test_sidebar_uses_text_content_for_preview():
    html = Path("static/index.html").read_text(encoding="utf-8")
    assert "preview.textContent = " in html
    assert "item.innerHTML = '<span>' + c.preview" not in html
