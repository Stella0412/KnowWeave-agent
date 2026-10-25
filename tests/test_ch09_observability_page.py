from pathlib import Path


def test_observability_page_is_safe_and_relative():
    html = Path("static/observability.html").read_text(encoding="utf-8")
    for token in [
        "/observability/api/summary",
        "/observability/api/turns",
        "/observability/api/tools",
        "textContent",
        "trace_id",
        "trace_url",
        "avg_tokens",
        "avg_duration_ms",
    ]:
        assert token in html
    assert "innerHTML" not in html
    assert "http://localhost" not in html
