"""静态评估页的回归断言：台账处置按钮必须能真正选中，且提交前选择可见。"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

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


def test_fetch_json_surfaces_http_errors():
    match = re.search(r"async function fetchJson\(url\) \{[\s\S]*?\n\}", HTML)
    assert match, "缺少 fetchJson 助手，接口失败会静默卡在「加载中…」"
    helper = match.group(0)

    node = shutil.which("node")
    if node is None:
        pytest.skip("node 不可用，跳过前端行为测试")

    script = (
        helper
        + "\nglobalThis.fetch = async () => ({ ok: false, status: 503,"
        + " json: async () => ({}) });"
        + "\nfetchJson('/rag-eval/api/latest')"
        + "\n  .then(() => console.log('resolved'))"
        + "\n  .catch((e) => console.log('rejected:' + e.message));"
    )
    completed = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, encoding="utf-8"
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "rejected:HTTP 503"


def test_init_reports_connection_failure_instead_of_hanging():
    init = re.search(r"\(async function init\(\) \{[\s\S]*?\n\}\)\(\);", HTML)
    assert init, "找不到 init()"
    body = init.group(0)
    assert "catch" in body, "init 失败时必须捕获异常"
    assert "无法连接服务" in body, "init 失败时必须给出可见提示"


# ---------------------------------------------------------------------------
# 飞轮趋势 + 置信度校准面板
# ---------------------------------------------------------------------------


def test_trend_panel_reads_the_trends_api():
    assert 'id="trend-panel"' in HTML
    assert "/rag-eval/api/trends" in HTML
    assert "make eval-flywheel" in HTML
    assert "TRIGGER=定时" in HTML
    # 同一评估集写在同一行，只有同集才比
    assert "同一评估集写在同一行" in HTML


def test_calibration_panel_reads_the_artifact_and_can_rerun():
    assert 'id="cal-panel"' in HTML
    assert "/rag-eval/api/calibration" in HTML
    assert "/rag-eval/api/calibrate" in HTML
    assert "make calibrate-confidence" in HTML
    assert "请重跑校准" in HTML  # 配置改了、产物对不上时要提示


def test_calibration_explains_weights_and_both_error_costs():
    assert "四信号加权" in HTML
    assert "0.5 / 有效证据数 0.2 / Top1-Top2 分差 0.2 / 关键条款命中 0.1" in HTML
    assert "可答被误拦" in HTML
    assert "误放" in HTML
    assert "飞轮的燃料" in HTML


def test_calibration_chart_has_selected_and_in_use_markers():
    assert "阈值 t" in HTML
    assert 'stroke-dasharray="5 4"' in HTML  # 选定阈值虚线
    assert "selected" not in HTML  # 图例用中文，不残留英文占位


def test_trend_delta_marks_direction_and_goodness():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node 不可用，跳过前端行为测试")
    match = re.search(r"function trendDelta\(delta, lowerIsBetter\) \{[\s\S]*?\n\}", HTML)
    assert match, "缺少 trendDelta，趋势表就看不出涨跌"
    script = match.group(0) + """
console.log([trendDelta(-0.05, false), trendDelta(0.05, false), trendDelta(-0.05, true)].join("|"));
"""
    completed = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, encoding="utf-8"
    )
    assert completed.returncode == 0, completed.stderr
    cells = completed.stdout.strip().split("|")
    assert "delta down" in cells[0] and "−5.0" in cells[0]
    assert "delta up" in cells[1] and "+5.0" in cells[1]
    # 误拦率下降是好事，别标成红的
    assert "delta up" in cells[2]


def _run_js(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node 不可用，跳过前端行为测试")
    completed = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, encoding="utf-8"
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _js_function(name):
    match = re.search(rf"function {name}\([^)]*\) \{{[\s\S]*?\n\}}", HTML)
    assert match, f"缺少 {name}"
    return match.group(0)


def _note_js(report):
    helpers = "\n".join(
        _js_function(name) for name in ("pct", "num", "nearestScanIndex")
    )
    return helpers + "\n" + _js_function("calibrationNote") + f"""
const report = {report};
console.log(calibrationNote(report));
"""


def test_calibration_note_says_no_change_when_thresholds_share_a_gap():
    # 在用 0.30 在推荐 0.33 左边（更宽松），但两线之间没有分数：不能写「往右挪」，
    # 更不能摆一条 98.8% → 98.8% 的假变化。
    report = {
        "in_use": 0.30, "recommended": 0.33, "youden": 0.9375,
        "at_in_use": {"pass_rate": 0.9875, "false_pass_rate": 0.05},
        "at_recommended": {"pass_rate": 0.9875, "false_pass_rate": 0.05},
        "scan": {
            "thresholds": [0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
            "pass_rates": [0.9875, 0.9875, 0.9875, 0.9875, 0.9875, 0.9875, 0.9875, 0.9875],
            "false_pass_rates": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0167],
        },
    }
    out = _run_js(_note_js(report))
    assert "更宽松" in out
    assert "指标完全一样" in out
    assert "往右" not in out


def test_calibration_note_moves_left_and_quotes_the_first_real_jump():
    # 往左找第一条误放率真的抬起来的线；中间平坦的 0.32/0.31 不要当成「升到 5.0%」。
    report = {
        "in_use": 0.30, "recommended": 0.33, "youden": 0.9,
        "at_in_use": {"pass_rate": 0.99, "false_pass_rate": 0.08},
        "at_recommended": {"pass_rate": 0.97, "false_pass_rate": 0.05},
        "scan": {
            "thresholds": [0.28, 0.29, 0.30, 0.31, 0.32, 0.33],
            "pass_rates": [0.99, 0.99, 0.99, 0.975, 0.97, 0.97],
            "false_pass_rates": [0.09, 0.08, 0.08, 0.05, 0.05, 0.05],
        },
    }
    out = _run_js(_note_js(report))
    assert "往左挪到在用值 0.30" in out
    assert "误放率 5.0% → 8.0%" in out  # 有真实变化时才列前后值
    # 从在用的 0.30 往左走，0.29 仍是 8.0%（没变），第一条真抬起来的是 0.28
    assert "再往左放到 0.28" in out and "9.0%" in out


def test_calibration_note_shows_what_zero_false_pass_costs():
    report = {
        "in_use": 0.30, "recommended": 0.33, "youden": 0.9,
        "at_in_use": {"pass_rate": 0.9875, "false_pass_rate": 0.05},
        "at_recommended": {"pass_rate": 0.9875, "false_pass_rate": 0.05},
        "scan": {
            "thresholds": [0.30, 0.33, 0.60, 0.66, 0.70],
            "pass_rates": [0.9875, 0.9875, 0.85, 0.8417, 0.80],
            "false_pass_rates": [0.05, 0.05, 0.0167, 0.0, 0.0],
        },
    }
    out = _run_js(_note_js(report))
    assert "压到 0" in out
    assert "0.66" in out and "84.2%" in out
