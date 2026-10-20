"""静态评估页的回归断言：台账处置按钮必须能真正选中，且提交前选择可见。"""

import re
from pathlib import Path

HTML = (
    Path(__file__).resolve().parents[1] / "static" / "rag_eval.html"
).read_text(encoding="utf-8")


def test_ledger_note_hidden_beats_display_flex():
    # .hidden(0,1,0) 与 .ledger-note(0,1,0) 同权重，后写的 display:flex 会赢，
    # 导致备注框永远展开、点处置按钮看不出任何反应。必须有一条更高权重的隐藏规则。
    assert re.search(r"\.ledger-note\.hidden\s*\{[^}]*display:\s*none", HTML), (
        "缺少 .ledger-note.hidden 规则，备注框会一直显示"
    )


def test_selected_action_has_visible_state():
    assert ".ledger-actions button.active" in HTML
    assert 'classList.toggle("active", b === btn)' in HTML


def test_submit_requires_an_explicit_choice():
    # 不允许把「没选处置方式就提交」默认当成真幻觉（已解决）
    assert 'faithPending[evalId] || "已解决"' not in HTML
    assert "请先选择处置方式" in HTML


def test_cancel_clears_selection():
    # 取消后不能残留选中态，否则下次提交会带上旧选择
    assert "delete faithPending[evalId]" in HTML
