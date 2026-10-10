from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, settings


def build_db_url(s: Settings | None = None) -> str:
    s = s or settings
    return (
        f"mysql+aiomysql://{s.mysql_user}:{s.mysql_password}"
        f"@{s.mysql_host}:{s.mysql_port}/{s.mysql_db}?charset=utf8mb4"
    )


def create_session_factory(url: str) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(url)
    return async_sessionmaker(engine, expire_on_commit=False)


engine = create_async_engine(build_db_url(settings))
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)
