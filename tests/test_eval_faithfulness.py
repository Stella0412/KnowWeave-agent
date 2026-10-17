import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_faithfulness", Path(__file__).resolve().parent.parent / "eval" / "eval_faithfulness.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_flags_number_plus_arrival_verb():
    ok, hits = mod.check_no_commitment("退款会在 3 个工作日内到账。")
    assert ok is False
    assert hits


def test_flags_specific_date_promise():
    ok, _ = mod.check_no_commitment("预计 10 月 20 日退款到您的账户。")
    assert ok is False


def test_allows_generic_no_commitment_wording():
    """库里那段政策本身就是正确措辞，不能误伤。"""
    ok, hits = mod.check_no_commitment(
        "退款原路退回，到账时效以平台售后规则为准，平台不承诺具体到账天数。"
    )
    assert ok is True
    assert hits == []


def test_allows_processing_time_not_arrival_time():
    """「3 个工作日处理」是处理时限不是到账承诺 —— 句中必须有到账类动词才算违规。"""
    ok, _ = mod.check_no_commitment("这个问题我们会在 3 个工作日内处理完毕。")
    assert ok is True


def test_allows_refusal_text():
    ok, _ = mod.check_no_commitment(mod.REFUSAL_TEXT)
    assert ok is True
