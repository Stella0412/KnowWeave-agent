from app.config import Settings
from app.core.llm import build_chat_model


def test_build_chat_model_uses_settings():
    s = Settings(llm_base_url="http://x/v1", llm_api_key="k", llm_model="m")
    m = build_chat_model(s)
    assert m.model_name == "m"
    assert m.openai_api_base == "http://x/v1"
    assert m.temperature == 0.7


def test_build_chat_model_can_pin_temperature_for_judging():
    s = Settings(llm_base_url="http://x/v1", llm_api_key="k", llm_model="m")
    assert build_chat_model(s, temperature=0.0).temperature == 0.0


def test_build_chat_model_retries_transient_connection_errors():
    """50 分钟的评估不能被一次 DNS 抖动打断；默认重试要够扛过短时断网。"""
    s = Settings(llm_base_url="http://x/v1", llm_api_key="k", llm_model="m")
    assert build_chat_model(s).max_retries == 8
    assert build_chat_model(s, max_retries=2).max_retries == 2


def test_build_structured_model_returns_runnable():
    from app.core.llm import build_structured_model
    from app.schemas.extract import AfterSaleExtraction

    s = Settings(llm_base_url="http://x/v1", llm_api_key="k", llm_model="m")
    r = build_structured_model(s, AfterSaleExtraction)
    assert hasattr(r, "ainvoke")
