"""Ch05 五条验收场景：对运行中的服务调 /chat/agent 与 /chat/ticket。

先启动：
  uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
再跑：
  uv run python eval/eval_ch05.py
"""

import asyncio
import os
import sys

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = os.getenv("KNOWWEAVE_BASE_URL", "http://127.0.0.1:8000")


async def ask(client, message, session_id=None):
    response = await client.post(
        "/chat/agent", json={"session_id": session_id, "message": message}
    )
    response.raise_for_status()
    return response.json()


def check(name, condition, detail=""):
    print(f"{'✓' if condition else '✗'} {name} {detail}")
    return condition


async def main():
    results = []
    async with httpx.AsyncClient(base_url=BASE, timeout=120.0) as client:
        policy = await ask(client, "退货政策是什么")
        results.append(check(
            "1. 政策问题强制检索",
            policy["trace"].get("forced_rag") is True,
            f"route={policy['route']} forced_rag={policy['trace'].get('forced_rag')}",
        ))

        logistics = await ask(client, "订单 1001 的物流到哪了")
        tools = [c["name"] for c in logistics.get("tool_calls", [])]
        results.append(check(
            "2. 业务问题 Agent 自调工具",
            "query_order" in tools and "query_logistics" in tools,
            f"tools={tools}",
        ))

        complaint = await ask(client, "我要投诉")
        action_types = [a["type"] for a in complaint.get("suggested_actions", [])]
        results.append(check(
            "3a. 投诉两个独立动作",
            action_types == ["transfer_human", "create_ticket"],
            f"actions={action_types}",
        ))
        ticket = await client.post(
            "/chat/ticket",
            json={
                "session_id": str(complaint["session_id"]),
                "ticket_type": "投诉",
                "description": "我要投诉",
            },
        )
        ticket.raise_for_status()
        ticket_no = ticket.json()["ticket_no"]
        results.append(check(
            "3b. 建工单写库",
            ticket_no.startswith("T"),
            f"ticket_no={ticket_no}",
        ))

        chitchat = await ask(client, "你好")
        results.append(check(
            "4. 闲聊固定话术",
            chitchat["route"] == "chitchat" and "客服知序" in (chitchat.get("answer") or ""),
            f"answer={chitchat.get('answer')}",
        ))

        complex_q = await ask(client, "订单 1001 的物流到哪了")
        complex_tools = [c["name"] for c in complex_q.get("tool_calls", [])]
        results.append(check(
            "5. ReAct 不止一步",
            len(complex_tools) > 1,
            f"tool_calls={complex_tools}",
        ))

    print(f"\n通过 {sum(results)}/{len(results)}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
