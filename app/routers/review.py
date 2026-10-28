from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.services.flywheel_service import ReviewRejected

router = APIRouter(prefix="/review", tags=["review"])
_VALID_STATUS = {"待处理", "已关闭", "已忽略"}
_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@router.get("", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "review.html")


class AnswerRequest(BaseModel):
    question: str | None = Field(default=None, max_length=4096)
    answer: str = Field(default="", max_length=16384)
    category: str | None = Field(default=None, max_length=255)
    action: Literal["new", "reuse", "update"] = "new"
    target_chunk_id: int | None = Field(default=None, gt=0)
    expected_signature: str | None = Field(default=None, min_length=64, max_length=64)


class ResolutionRequest(BaseModel):
    resolution: str = Field(min_length=1, max_length=500)


def _service(request: Request):
    service = getattr(request.app.state, "flywheel", None)
    if service is None:
        raise HTTPException(status_code=503, detail="飞轮服务未初始化")
    return service


def _rejected(exc: ReviewRejected) -> JSONResponse:
    content = {"detail": exc.detail, **exc.extra}
    return JSONResponse(status_code=exc.status_code, content=content)


@router.get("/api/queue")
async def queue(
    request: Request, status: str = "待处理", limit: int = 50, offset: int = 0,
) -> dict:
    if status not in _VALID_STATUS:
        raise HTTPException(status_code=400, detail="未知队列状态")
    if not 1 <= limit <= 200 or offset < 0:
        raise HTTPException(status_code=400, detail="limit 1–200，offset >= 0")
    try:
        items, total = await _service(request).list_queue(
            status=status, limit=limit, offset=offset
        )
    except ReviewRejected as exc:
        return _rejected(exc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"队列不可用：{exc}") from exc
    return {
        "items": items, "total": total,
        "status": status, "limit": limit, "offset": offset,
    }


@router.get("/api/clusters/{cluster_id}")
async def detail(request: Request, cluster_id: int) -> dict:
    try:
        return await _service(request).cluster_detail(cluster_id)
    except ReviewRejected as exc:
        return _rejected(exc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"详情不可用：{exc}") from exc


@router.post("/api/clusters/{cluster_id}/answer")
async def answer(request: Request, cluster_id: int, req: AnswerRequest):
    try:
        result = await _service(request).submit_answer(
            cluster_id, question=req.question, answer=req.answer, category=req.category,
            action=req.action, target_chunk_id=req.target_chunk_id,
            expected_signature=req.expected_signature,
        )
    except ReviewRejected as exc:
        return _rejected(exc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"补答失败：{exc}") from exc
    return JSONResponse(
        status_code=200 if result["vectorized"] else 202, content=result
    )


@router.post("/api/clusters/{cluster_id}/suggestion")
async def suggestion(request: Request, cluster_id: int):
    pipeline = getattr(request.app.state, "suggestions", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="AI 建议服务未初始化")
    try:
        result = await pipeline.generate(cluster_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="AI 建议暂不可用，请稍后重试") from exc
    if result is None:
        raise HTTPException(status_code=409, detail="仅待处理的主问题可生成建议")
    return _service(request)._suggestion_fields(result)


@router.post("/api/clusters/{cluster_id}/ignore")
async def ignore(request: Request, cluster_id: int, req: ResolutionRequest):
    try:
        return await _service(request).ignore(cluster_id, req.resolution)
    except ReviewRejected as exc:
        return _rejected(exc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"驳回失败：{exc}") from exc


@router.post("/api/clusters/{cluster_id}/reopen")
async def reopen(request: Request, cluster_id: int, req: ResolutionRequest):
    try:
        return await _service(request).reopen(cluster_id, req.resolution)
    except ReviewRejected as exc:
        return _rejected(exc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"重开失败：{exc}") from exc


@router.post("/api/vectorize")
async def vectorize(request: Request):
    try:
        return await _service(request).vectorize_pending()
    except ReviewRejected as exc:
        return _rejected(exc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"向量化不可用：{exc}") from exc


@router.get("/api/metrics")
async def metrics(request: Request) -> dict:
    try:
        return await _service(request).metrics()
    except ReviewRejected as exc:
        return _rejected(exc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"指标不可用：{exc}") from exc
