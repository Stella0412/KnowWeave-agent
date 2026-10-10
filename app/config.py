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


settings = Settings()
