from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import Settings, settings
from app.db.base import Base
from app.db.engine import engine
from app.graph.build import build_graph
from app.graph.runtime import open_checkpointer
from app.routers import chat, extract, rag_eval

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def validate_config(s: Settings | None = None) -> None:
    s = s or settings
    if not s.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY，请在 .env 中配置后重启")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"[警告] 数据库不可用，跳过建表: {e}")

    async with open_checkpointer(settings.langgraph_checkpoint_path) as saver:
        app.state.graph = build_graph(
            checkpointer=saver,
            max_agent_steps=settings.max_agent_steps,
        )
        yield


app = FastAPI(title="知序智能客服", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(extract.router)
app.include_router(rag_eval.router)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
