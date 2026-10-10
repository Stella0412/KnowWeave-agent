from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import Settings, settings
from app.routers import chat, extract

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def validate_config(s: Settings | None = None) -> None:
    s = s or settings
    if not s.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY，请在 .env 中配置后重启")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    yield


app = FastAPI(title="知序智能客服", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(extract.router)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
