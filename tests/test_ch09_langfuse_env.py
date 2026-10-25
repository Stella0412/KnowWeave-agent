from pathlib import Path

from app.jobs.init_langfuse_env import ensure_langfuse_env, main


def test_main_fills_env_without_overwriting_or_printing_secrets(tmp_path, capsys):
    path = tmp_path / ".env"
    path.write_text(
        "LLM_API_KEY=keep-me\nLANGFUSE_PUBLIC_KEY=existing-public\n",
        encoding="utf-8",
    )
    main(path)
    text = path.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "LLM_API_KEY=keep-me" in text
    assert "LANGFUSE_PUBLIC_KEY=existing-public" in text
    assert "LANGFUSE_SECRET_KEY=sk-lf-local-" in text
    assert "LANGFUSE_NEXTAUTH_SECRET=" in text
    assert "LANGFUSE_INIT_PROJECT_NAME=知序" in text
    assert "值未打印" in out
    assert "existing-public" not in out
    assert "sk-lf-local-" not in out


def test_ensure_langfuse_env_is_silent_and_counts(tmp_path, capsys):
    path = tmp_path / ".env"
    path.write_text("LANGFUSE_PUBLIC_KEY=existing-public\n", encoding="utf-8")
    n = ensure_langfuse_env(path)
    out = capsys.readouterr().out
    assert n > 0
    assert out == ""
