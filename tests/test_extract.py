import httpx
import pytest
from fastapi import FastAPI

from app.routers import extract as extract_module
from app.schemas.extract import AfterSaleExtraction
from app.services.extract_service import ExtractService


class FakeStructuredModel:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, messages):
        return self._result


@pytest.mark.asyncio
async def test_extract_service_returns_parsed():
    expected = AfterSaleExtraction(
        order_id="88231", claim_type="退货退款", expected_resolution="退货退款"
    )
    svc = ExtractService(FakeStructuredModel(expected))
    got = await svc.extract("订单号88231 想退货退款")
    assert got == expected


def test_extraction_allows_null_order_id():
    got = AfterSaleExtraction(order_id=None, claim_type="仅退款", expected_resolution="退款")
    assert got.order_id is None
    assert got.model_dump()["order_id"] is None


async def test_extract_router_empty_text_422():
    app = FastAPI()
    app.include_router(extract_module.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/extract", json={"text": ""})
    assert r.status_code == 422


async def test_extract_router_502_on_model_error(monkeypatch):
    class Boom:
        async def extract(self, text):
            raise RuntimeError("upstream down")

    monkeypatch.setattr(extract_module, "_extract_service", Boom())
    app = FastAPI()
    app.include_router(extract_module.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/extract", json={"text": "你好"})
    assert r.status_code == 502
