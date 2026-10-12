from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    token_budget: int = 4096

    # MySQL（Docker 起，连接参数从 .env 读）
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "knowweave123"
    mysql_db: str = "knowweave"
    # 工具执行
    tool_timeout: float = 5.0
    tool_retries: int = 1

    # 嵌入模型（BGE-M3 via 硅基流动 OpenAI 兼容端点）
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str = ""
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024

    # Milvus：默认 Lite 内嵌（.db 文件），改 http://127.0.0.1:19530 切 standalone
    milvus_uri: str = "data/milvus_knowledge.db"
    milvus_collection: str = "knowledge"

    # 在线检索（dense 单路 Top-K + 相似度阈值）
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.55

    # 切分（四项：目标 / 硬上限 / 重叠 / 表格单块最大行数）
    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 80
    chunk_table_max_rows: int = 30

    # 建库 / 挖知识
    kb_docs_dir: str = "data/kb"
    mine_batch_size: int = 20


settings = Settings()
