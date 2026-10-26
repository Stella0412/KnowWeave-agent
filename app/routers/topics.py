from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.schemas.topics import DistributionResponse

router = APIRouter(prefix="/topics", tags=["topics"])
_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


def _service(request: Request):
    service = getattr(request.app.state, "topic_admin", None)
    if service is None:
        raise HTTPException(status_code=503, detail="主题服务未初始化")
    return service


@router.get("", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "topics.html")


@router.get("/api/distribution", response_model=DistributionResponse)
async def distribution(request: Request, days: int = 30, weighted: bool = False):
    if not 1 <= days <= 365:
        raise HTTPException(status_code=400, detail="days 1–365")
    try:
        return await _service(request).distribution(days=days, weighted=weighted)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"分布不可用：{exc}") from exc


@router.get("/api/questions")
async def questions(request: Request, label: str, limit: int = 20):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="limit 1–100")
    try:
        return await _service(request).questions(label, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"问题样例不可用：{exc}") from exc
