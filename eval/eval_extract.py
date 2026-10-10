import asyncio
import json

from app.config import settings
from app.core.llm import build_structured_model
from app.schemas.extract import AfterSaleExtraction
from app.services.extract_service import ExtractService


async def main():
    svc = ExtractService(build_structured_model(settings, AfterSaleExtraction))
    with open("eval/samples_extract.jsonl", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    order_ok = claim_ok = total = 0
    for s in samples:
        total += 1
        got = await svc.extract(s["text"])
        if got.order_id == s["order_id"]:
            order_ok += 1
        if got.claim_type == s["claim_type"]:
            claim_ok += 1
        print(
            f"[{s['text'][:12]}…] 期望 order={s['order_id']} claim={s['claim_type']} "
            f"实际 order={got.order_id} claim={got.claim_type} resolution={got.expected_resolution}"
        )

    print(f"\n订单号准确率: {order_ok}/{total}")
    print(f"诉求类型准确率: {claim_ok}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
