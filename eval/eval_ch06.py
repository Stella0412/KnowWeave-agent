"""端到端验收：需先 uv run uvicorn app.main:app --port 8000。"""
import asyncio
import sys

import httpx

BASE = "http://127.0.0.1:8000"


async def chat(client, session_id, message):
    resp = await client.post("/chat/agent", json={"session_id": session_id, "message": message})
    resp.raise_for_status()
    return resp.json()


async def main() -> int:
    failures = []
    async with httpx.AsyncClient(base_url=BASE, timeout=180) as client:
        # 1. 多轮意图 + 指代
        first = await chat(client, None, "订单1001的猫爬架什么时候到")
        sid = str(first["session_id"])
        second = await chat(client, sid, "它能退吗")
        if second.get("intent") != "refund_return" or "猫爬架" not in (second.get("trace", {}).get("resolved_query") or ""):
            failures.append("acceptance-1")
        print("1", second.get("intent"), second.get("trace", {}).get("resolved_query"))

        # 2. 怪问题落 other
        odd = await chat(client, None, "你们总部在哪")
        if odd.get("intent") != "other" or odd.get("route") != "clarify":
            failures.append("acceptance-2")
        print("2", odd.get("intent"), odd.get("route"))

        # 3. 先给订单再问「这个能退吗」
        ctx = await chat(client, None, "订单1001的猫爬架")
        sid = str(ctx["session_id"])
        third = await chat(client, sid, "这个能退吗")
        trace = third.get("trace", {})
        if third.get("route") != "refund_flow" or trace.get("order_no") != "1001" or not trace.get("policy_queries"):
            failures.append("acceptance-3")
        print("3", trace.get("resolved_query"), trace.get("order_no"), trace.get("flow_steps"))

        # 4. 不带订单号 -> 选择器 -> resume
        ask = await chat(client, None, "我要退款")
        if not ask.get("interrupted"):
            failures.append("acceptance-4-interrupt")
        sid = str(ask["session_id"])
        resumed = await client.post("/chat/agent/resume", json={"session_id": sid, "order_no": "1001"})
        resumed.raise_for_status()
        final = resumed.json()
        if final.get("interrupted") or not any(a.get("type") == "refund_form" for a in final.get("suggested_actions", [])):
            failures.append("acceptance-4-resume")
        print("4", final.get("answer"), [a.get("type") for a in final.get("suggested_actions", [])])

    print(f"\n端到端验收: {'PASS' if not failures else 'FAIL ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
