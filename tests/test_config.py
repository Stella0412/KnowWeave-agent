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
