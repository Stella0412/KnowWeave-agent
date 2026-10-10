from app.config import Settings


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.llm_base_url == "https://api.deepseek.com"
    assert s.llm_model == "deepseek-chat"
    assert s.token_budget == 4096


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("TOKEN_BUDGET", "2048")
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_model == "qwen2.5"
    assert s.llm_api_key == "sk-test"
    assert s.token_budget == 2048
