from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services import rag_eval_service as service

router = APIRouter(prefix="/rag-eval", tags=["rag-eval"])

_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


class RunRequest(BaseModel):
    mode: str = "retrieval"


class FaithCaseRequest(BaseModel):
    status: str
    resolution: str | None = None


@router.get("", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "rag_eval.html")


@router.get("/api/latest")
def latest() -> dict:
    return {"latest": service.read_latest()}


@router.get("/api/status")
def status() -> dict:
    return service.get_status()


@router.post("/api/run", status_code=202)
def run(req: RunRequest) -> dict:
    if req.mode not in service.VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"未知评估模式：{req.mode}（可选：{'/'.join(service.VALID_MODES)}）",
        )
    try:
        accepted, state = service.start_run(req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not accepted:
        raise HTTPException(status_code=409, detail="已有评估在运行，请等待完成")
    return state


@router.get("/api/faith-cases")
async def faith_cases() -> dict:
    try:
        return await service.list_faith_cases()
    except Exception as e:  # noqa: BLE001 —— 台账依赖 MySQL，挂了也要让页面看到原因
        raise HTTPException(status_code=503, detail=f"台账不可用：{e}")


@router.post("/api/faith-cases/{eval_id}")
async def update_faith_case(eval_id: str, req: FaithCaseRequest) -> dict:
    try:
        case = await service.resolve_faith_case(eval_id, req.status, req.resolution)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if case is None:
        raise HTTPException(status_code=404, detail="台账里没有这道题")
    return {"ok": True, "case": case}
