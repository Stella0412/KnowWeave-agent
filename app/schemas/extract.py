from typing import Literal

from pydantic import BaseModel, Field

ClaimType = Literal["退货退款", "换货", "仅退款", "维修", "物流查询", "其他"]


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)


class AfterSaleExtraction(BaseModel):
    """售后信息提取结果。"""

    order_id: str | None = Field(default=None, description="订单号，用户未提供则为 null")
    claim_type: ClaimType
    expected_resolution: str = Field(description="用户期望的解决方案，一句话自然语言")
