from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.llm import build_structured_model
from app.schemas.extract import AfterSaleExtraction, ExtractRequest
from app.services.extract_service import ExtractService

router = APIRouter(prefix="/extract", tags=["extract"])

# 懒实例化：避免 import 时用空 key 构造 ChatOpenAI 报错（Risk A）。
# 单例缓存在模块级，首次请求时才真正构建。
_extract_service: ExtractService | None = None


def _get_extract_service() -> ExtractService:
    global _extract_service
    if _extract_service is None:
        _extract_service = ExtractService(build_structured_model(settings, AfterSaleExtraction))
    return _extract_service


@router.post("", response_model=AfterSaleExtraction)
async def extract(req: ExtractRequest):
    try:
        return await _get_extract_service().extract(req.text)
    except Exception as e:  # 上游模型报错 → 502，不静默
        raise HTTPException(status_code=502, detail=f"模型调用失败: {e}")
