from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    token_budget: int = 4096
    # ch05 图编排
    langgraph_checkpoint_path: str = "data/langgraph_checkpoints.sqlite"
    max_agent_steps: int = 6

    # ch06 意图识别：默认大模型判；小模型升级只留配置种子，本章不实现
    intent_model: str = ""
    intent_small_model: str = ""
    intent_mode: str = "accuracy"
    intent_conf_threshold: float = 0.7

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
    # 仓库 .env / .env.example 用的是 EMBEDDING_*（沿用仓库既有约定），与字段名 embed_* 不同名 →
    # 显式给 alias，否则会静默读到空 key（EMBEDDING_API_KEY 匹配不上 embed_api_key）。
    embed_base_url: str = Field(
        "https://api.siliconflow.cn/v1",
        validation_alias=AliasChoices("embed_base_url", "EMBEDDING_BASE_URL", "EMBED_BASE_URL"),
    )
    embed_api_key: str = Field(
        "",
        validation_alias=AliasChoices("embed_api_key", "EMBEDDING_API_KEY", "EMBED_API_KEY"),
    )
    embed_model: str = Field(
        "BAAI/bge-m3",
        validation_alias=AliasChoices("embed_model", "EMBEDDING_MODEL", "EMBED_MODEL"),
    )
    embed_dim: int = 1024

    # Milvus：默认 Lite 内嵌（.db 文件），改 http://127.0.0.1:19530 切 standalone
    milvus_uri: str = "data/milvus_knowledge.db"
    milvus_collection: str = "knowledge"

    # 在线检索（dense 单路 Top-K + 相似度阈值）
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.55

    # ch04 混合检索与重排（spec §4）
    recall_top_k: int = 50          # 每路召回条数
    rrf_k: int = 60                 # RRF 平滑常数
    rerank_top_n: int = Field(
        10, validation_alias=AliasChoices("RERANK_TOP_K", "rerank_top_n")
    )  # 精排后进 prompt 的条数；RERANK_TOP_K 优先
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # 留空即复用 embed_base_url / embed_api_key（沿用仓库既有命名约定，故给 alias）
    rerank_base_url: str = Field(
        "", validation_alias=AliasChoices("rerank_base_url", "RERANK_BASE_URL")
    )
    rerank_api_key: str = Field(
        "", validation_alias=AliasChoices("rerank_api_key", "RERANK_API_KEY")
    )
    rerank_timeout: float = 20.0
    rerank_min_score: float = 0.25  # 拒答闸门阈值（2026-10-17 用 39 题评估集在 chunk 粒度复标）
    rag_strategy: str = "hybrid_rerank"   # 生产策略
    rag_tool_timeout: float = 30.0  # query_faq 专用工具超时（见 Task 15 说明）

    # 切分（四项：目标 / 硬上限 / 重叠 / 表格单块最大行数）
    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 80
    chunk_table_max_rows: int = 30

    # 建库 / 挖知识
    kb_docs_dir: str = "data/kb"
    mine_batch_size: int = 20

    # ch07 上下文管理
    model_context_window: int = 64000
    max_output_tokens: int = 2000
    max_user_input_tokens: int = 2000
    tool_result_max_tokens: int = 1200
    system_prefix_reserve_tokens: int = 1500
    evidence_tokens_per_chunk: int = 200
    summary_inject_max_tokens: int = 1000
    context_safety_margin_tokens: int = 1250
    context_target_rounds: int = 20
    layer1_share: float = 0.7
    steady_turn_tokens: int = 1200
    classifier_history_max_tokens: int = 1200
    layer2_assistant_prefix_chars: int = 60
    order_json_max_tokens: int = 600
    cjk_tokens_per_char: float = 0.6
    token_estimate_factor: float = 1.15
    summary_input_max_tokens: int = 6000
    summary_message_max_tokens: int = 800
    summary_tool_result_max_tokens: int = 400
    summary_max_chars: int = 200
    summary_max_output_tokens: int = 400
    summary_retry_cooldown_seconds: int = 60
    log_file: str = "log/app.log"


settings = Settings()
