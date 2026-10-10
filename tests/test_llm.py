from app.config import Settings
from app.core.llm import build_chat_model


def test_build_chat_model_uses_settings():
    s = Settings(llm_base_url="http://x/v1", llm_api_key="k", llm_model="m")
    m = build_chat_model(s)
    assert m.model_name == "m"
    assert m.openai_api_base == "http://x/v1"
