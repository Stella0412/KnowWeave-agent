from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter(prefix="/observability", tags=["observability"])

_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
_VALID_DAYS = {1, 7, 30}
_VALID_STATUS = {"ok", "error", "interrupted", "cancelled"}


def _service(request: Request):
    service = getattr(request.app.state, "observability_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="观测服务未初始化")
    return service


@router.get("", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "observability.html")


@router.get("/api/summary")
async def summary(request: Request, days: int = 7) -> dict:
    if days not in _VALID_DAYS:
        raise HTTPException(status_code=400, detail="days 只能是 1/7/30")
    return await _service(request).summary(days)


@router.get("/api/turns")
async def turns(
    request: Request,
    days: int = 7,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    route: str | None = None,
) -> dict:
    if days not in _VALID_DAYS:
        raise HTTPException(status_code=400, detail="days 只能是 1/7/30")
    if status is not None and status not in _VALID_STATUS:
        raise HTTPException(status_code=400, detail="未知轮次状态")
    if not 1 <= limit <= 200 or offset < 0:
        raise HTTPException(status_code=400, detail="limit 1–200，offset >= 0")
    return await _service(request).turns(days, limit, offset, status=status, route=route)


@router.get("/api/tools")
async def tools(request: Request, days: int = 7) -> dict:
    if days not in _VALID_DAYS:
        raise HTTPException(status_code=400, detail="days 只能是 1/7/30")
    return await _service(request).tools(days)
