import asyncio
import json
import sys
from pathlib import Path

# 允许 `python eval/eval_tools.py` 直接运行：脚本目录默认进 sys.path[0]，app 包在仓库根
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Windows 控制台默认 GBK，强制 UTF-8 输出避免 ✓/✗ 报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.core.llm import build_chat_model
from app.db.engine import create_session_factory
from app.prompts.customer_service import SYSTEM_PROMPT
from app.tools.builtin import register_builtins
from app.tools.registry import ToolRegistry


async def main():
    model = build_chat_model(settings)
    sf = create_session_factory("sqlite+aiosqlite:///:memory:")
    registry = ToolRegistry()
    register_builtins(registry)
    tools = [
        {"name": d.name, "description": d.description, "parameters": d.parameters}
        for d in registry.snapshot().values()
    ]
    bound = model.bind_tools(tools)
    with open("eval/samples_tools.jsonl", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    ok = 0
    for s in samples:
        ai = await bound.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=s["text"])]
        )
        picked = ai.tool_calls[0]["name"] if ai.tool_calls else None
        hit = picked == s["tool"]
        ok += int(hit)
        print(f"[{s['text'][:14]}…] 期望={s['tool']} 实际={picked} {'✓' if hit else '✗'}")

    print(f"\n工具选择准确率: {ok}/{len(samples)}")


if __name__ == "__main__":
    asyncio.run(main())
