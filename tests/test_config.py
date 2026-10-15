import os

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def _clean_config_env(monkeypatch):
    """断言「默认值」的用例必须环境干净。

    pymilvus.settings 在 import 时会 `load_dotenv()`，把仓库 `.env` 注入进程 os.environ
    （第三方副作用，见 SDD Task 9 Ruling）；pytest 收集阶段就会 import 到它，于是 `Settings(_env_file=None)`
    读到的仍是本机 .env 的值，「默认值」断言随本机配置漂移。故本文件先清掉相关环境变量。
    """
    for key in list(os.environ):
        if key.startswith(("MYSQL_", "TOOL_", "LLM_", "TOKEN_", "EMBED", "MILVUS_", "RERANK_", "RAG_")):
            monkeypatch.delenv(key, raising=False)


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


def test_settings_mysql_defaults():
    s = Settings(_env_file=None)
    assert s.mysql_host == "127.0.0.1"
    assert s.mysql_port == 3306
    assert s.mysql_user == "root"
    assert s.mysql_db == "knowweave"
    assert s.tool_timeout == 5.0
    assert s.tool_retries == 1


def test_embedding_and_milvus_defaults():
    s = Settings(_env_file=None)
    assert s.embed_base_url == "https://api.siliconflow.cn/v1"
    assert s.embed_api_key == ""
    assert s.embed_model == "BAAI/bge-m3"
    assert s.embed_dim == 1024
    assert s.milvus_uri == "data/milvus_knowledge.db"
    assert s.milvus_collection == "knowledge"
    assert s.retrieval_top_k == 5
    # 预标定值：实测相关 cos≈0.78、不相关≈0.45；Task 15 用标注样例最终标定后回写
    assert 0.0 < s.retrieval_min_score < 1.0
    assert s.chunk_target_tokens == 500
    assert s.chunk_max_tokens == 800
    assert s.chunk_overlap_tokens == 80
    assert s.chunk_table_max_rows == 30
    assert s.kb_docs_dir == "data/kb"
    assert s.mine_batch_size == 20


def test_embedding_settings_read_repo_env_names(monkeypatch):
    """嵌入三项必须读仓库 `.env` / `.env.example` 既有的 EMBEDDING_* 命名（沿用仓库既有约定）。

    字段名是 embed_*（→ 默认认 EMBED_*），与 .env 里的 EMBEDDING_* 对不上就会静默读到空 key。
    """
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    s = Settings(_env_file=None)
    assert s.embed_base_url == "https://example.invalid/v1"
    assert s.embed_api_key == "sk-test"
    assert s.embed_model == "BAAI/bge-m3"


def test_embedding_settings_still_accept_kwargs():
    """字段名 kwargs 仍可用（embed_* 加了 alias 后不能被锁死）。"""
    s = Settings(_env_file=None, embed_api_key="sk-kw", embed_model="m")
    assert s.embed_api_key == "sk-kw"
    assert s.embed_model == "m"


def test_settings_mysql_from_env(monkeypatch):
    monkeypatch.setenv("MYSQL_HOST", "10.0.0.9")
    monkeypatch.setenv("MYSQL_PORT", "3307")
    monkeypatch.setenv("MYSQL_PASSWORD", "p@ss")
    monkeypatch.setenv("TOOL_TIMEOUT", "8")
    s = Settings(_env_file=None)
    assert s.mysql_host == "10.0.0.9"
    assert s.mysql_port == 3307
    assert s.mysql_password == "p@ss"
    assert s.tool_timeout == 8.0


def test_ch04_rag_defaults():
    s = Settings(_env_file=None)
    assert s.recall_top_k == 50
    assert s.rrf_k == 60
    assert s.rerank_top_n == 10
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank_base_url == ""      # 空 = 复用 embed_base_url
    assert s.rerank_api_key == ""       # 空 = 复用 embed_api_key
    assert s.rerank_min_score == 0.10
    assert s.rag_strategy == "hybrid_rerank"
    # ch03 的 dense 单路参数必须保留（纯向量策略与 ch03 的 eval_retrieval.py 仍用）
    assert s.retrieval_top_k == 5
    assert 0.0 < s.retrieval_min_score < 1.0


def test_rerank_settings_read_repo_env_names(monkeypatch):
    monkeypatch.setenv("RERANK_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("RERANK_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    assert s.rerank_base_url == "https://example.invalid/v1"
    assert s.rerank_api_key == "sk-test"
