from pathlib import Path


def test_frontend_has_actions_and_ticket_modal():
    html = Path("static/index.html").read_text(encoding="utf-8")
    markers = [
        "case 'actions'",
        "renderActions",
        "transfer_human",
        "openTicketModal",
        "已转接人工客服",
        "工单已创建",
        "fetch('/chat/ticket'",
        "ticketType",
        "ticketDesc",
        "Escape",
    ]
    for marker in markers:
        assert marker in html, marker
