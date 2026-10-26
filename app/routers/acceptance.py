from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter(prefix="/acceptance", tags=["acceptance"])
_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


def _service(request: Request):
    service = getattr(request.app.state, "acceptance", None)
    if service is None:
        raise HTTPException(status_code=503, detail="验收服务未初始化")
    return service


def _page(name: str) -> FileResponse:
    return FileResponse(_STATIC_DIR / name)


@router.get("", include_in_schema=False)
def page() -> FileResponse:
    return _page("acceptance.html")


@router.get("/eval", include_in_schema=False)
def eval_page() -> FileResponse:
    return _page("acceptance_eval.html")


@router.get("/data", include_in_schema=False)
def data_page() -> FileResponse:
    return _page("acceptance_data.html")


@router.get("/errors", include_in_schema=False)
def errors_page() -> FileResponse:
    return _page("acceptance_errors.html")


@router.get("/api/overview")
async def overview(request: Request):
    return _service(request).overview()


@router.get("/api/eval")
async def eval_detail(request: Request):
    return _service(request).eval_detail()


@router.get("/api/data")
async def artifacts(request: Request):
    return {"artifacts": _service(request).artifacts()}


@router.get("/api/errors")
async def errors(request: Request, limit: int = 50):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=400, detail="limit 1–500")
    return {"errors": _service(request).errors(limit)}
