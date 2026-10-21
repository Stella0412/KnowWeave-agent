from app.config import Settings
from app.core.llm import build_chat_model


def test_intent_config_defaults():
    s = Settings(_env_file=None)
    assert s.intent_model == ""
    assert s.intent_small_model == ""
    assert s.intent_mode == "accuracy"
    assert s.intent_conf_threshold == 0.7


def test_chat_model_accepts_model_name():
    s = Settings(_env_file=None, llm_api_key="test-key")
    model = build_chat_model(s, model_name="intent-large")
    assert model.model_name == "intent-large"
