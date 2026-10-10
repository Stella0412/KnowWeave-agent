import asyncio
import json

from app.config import settings
from app.core.llm import build_chat_model
from app.services.chat_service import ChatService
from app.services.context_service import ContextStore


async def main():
    svc = ChatService(build_chat_model(settings), ContextStore(settings.token_budget))
    with open("eval/samples_chat.jsonl", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    for i, s in enumerate(samples):
        parts = []
        async for e in svc.stream(f"eval-{i}", s["text"]):
            if e["type"] == "token":
                parts.append(e["content"])
        reply = "".join(parts)
        print(f"问: {s['text']}\n约束: {s['constraint']}\n答: {reply}\n---")


if __name__ == "__main__":
    asyncio.run(main())
