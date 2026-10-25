import json

from app.config import Settings
from app.observability.prices import PriceBook, load_prices


def test_ch09_settings_defaults():
    s = Settings(_env_file=None)
    assert s.langfuse_base_url == "http://localhost:3000"
    assert s.langfuse_tracing_enabled is True
    assert s.model_prices_path == "config/model_prices.json"
    assert s.low_conf_merge_threshold == 0.92
    assert s.low_conf_merge_limit == 2000
    assert s.low_conf_merge_batch_size == 64


def test_prices_load_null_as_unpriced(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"models": {
        "deepseek-chat": {"input_per_million": None, "output_per_million": None},
        "priced-model": {"input_per_million": 1.0, "output_per_million": 2.0},
    }}), encoding="utf-8")
    book = load_prices(path)
    assert book.get("deepseek-chat") is None
    assert book.cost("priced-model", 1_000_000, 1_000_000) == 3.0
    assert book.cost("priced-model", 500_000, 0) == 0.5


def test_prices_bad_file_degrades_to_empty(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text("{bad json", encoding="utf-8")
    assert load_prices(path) == PriceBook({})


def test_prices_negative_is_ignored(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"models": {
        "bad": {"input_per_million": -1, "output_per_million": 1},
    }}), encoding="utf-8")
    assert load_prices(path).get("bad") is None
