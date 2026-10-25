from pathlib import Path


def test_review_page_is_safe_and_wired():
    html = Path("static/review.html").read_text(encoding="utf-8")
    for token in [
        "/review/api/queue", "/review/api/metrics",
        "/answer", "/ignore", "/reopen", "/vectorize",
        "has_snippets", "missing", "empty",
        "redirected_from", "occurrences", "variants",
    ]:
        assert token in html
    assert "textContent" in html
    assert "innerHTML" not in html
    assert "http://localhost" not in html
