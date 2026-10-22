import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import Settings, settings
from app.core.logging import configure_context_logging
from app.db.base import Base
from app.db.engine import SessionFactory, engine
from app.db.seed import seed_orders
from app.graph.build import build_graph
from app.graph.runtime import open_checkpointer
from app.routers import chat, conversations, extract, rag_eval
from app.services.chat_service import build_chat_service
from app.services.context_manager import log_startup_budget
from app.services.token_counter import TokenCounter

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# uvicorn 只配置自己的 logger，应用日志默认被根 logger 的 WARNING 挡住；
# 编排 trace 是 ch05 的验收口径（forced_rag / tools / stop_reason），必须能看到。
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("knowweave.turn").setLevel(logging.INFO)


def validate_config(s: Settings | None = None) -> None:
    s = s or settings
    if not s.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY，请在 .env 中配置后重启")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    context_logger = configure_context_logging(settings.log_file)
    counter = TokenCounter(
        cjk_tokens_per_char=settings.cjk_tokens_per_char,
        estimate_factor=settings.token_estimate_factor,
    )
    log_startup_budget(context_logger, settings, counter)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await seed_orders(SessionFactory)
    except Exception as e:
        print(f"[警告] 数据库不可用，跳过建表: {e}")

    async with open_checkpointer(settings.langgraph_checkpoint_path) as saver:
        app.state.graph = build_graph(
            checkpointer=saver,
            max_agent_steps=settings.max_agent_steps,
        )
        app.state.chat_service = build_chat_service(
            app.state.graph, settings=settings, session_factory=SessionFactory,
        )
        try:
            yield
        finally:
            await app.state.chat_service.shutdown()


app = FastAPI(title="知序智能客服", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(extract.router)
app.include_router(rag_eval.router)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
