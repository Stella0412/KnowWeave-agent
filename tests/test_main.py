import pytest

from app.config import Settings
from app.main import validate_config


def test_validate_config_raises_without_key():
    with pytest.raises(RuntimeError):
        validate_config(Settings(llm_api_key=""))


def test_validate_config_passes_with_key():
    validate_config(Settings(llm_api_key="sk-1"))  # 不抛即通过


def test_main_app_imports():
    import app.main  # noqa: F401  # 组装不抛即通过
