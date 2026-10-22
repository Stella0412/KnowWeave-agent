# eval/eval_ch07_context.py
"""真实服务验收：默认 20 轮不触发；演示配置出现降级/摘要级联。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOG_PATH = Path("log/app.log")


def _events():
    """读取当前日志及轮转备份（.1/.2/.3），解析全部 JSON 事件。"""
    paths = [LOG_PATH]
    for suffix in (".1", ".2", ".3"):
        rotated = Path(str(LOG_PATH) + suffix)
        if rotated.exists():
            paths.append(rotated)
    items = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                items.append(payload)
    return items


def _events_for(session_id: int):
    """按本轮会话 id 过滤事件，避免 10MB 日志轮转把事件拆到备份文件后漏判。"""
    return [e for e in _events() if e.get("conversation_id") == session_id]


def _pad(text: str, min_chars: int) -> str:
    filler = "这是用于上下文预算验收的填充内容。"
    while len(text) < min_chars:
        text += filler
    return text


async def _ask(client, base_url, session_id, message):
    resp = await client.post(
        f"{base_url}/chat/agent",
        json={"session_id": session_id, "message": message},
    )
    resp.raise_for_status()
    return resp.json()


async def run(mode: str, turns: int, base_url: str) -> int:
    async with httpx.AsyncClient(timeout=120.0) as client:
        session_id = None
        # 默认每轮控制在约 1000 token，留出 assistant 开销后仍 <= 1200；
        # 演示每轮更长，确保 20 轮内出现级联。
        pad = 900 if mode == "default" else 1600
        for index in range(turns):
            message = _pad(f"第{index}轮，订单1001的猫爬架进度怎么样？", pad)
            payload = await _ask(client, base_url, session_id, message)
            session_id = str(payload["session_id"])
            print(
                f"turn {index + 1}: session={session_id} "
                f"interrupted={payload.get('interrupted')}"
            )
        events = _events_for(int(session_id))
        if mode == "default":
            demoted = [e for e in events if e.get("event") == "layer1 降级"]
            triggered = [e for e in events if e.get("event") == "summary trigger"]
            assert not demoted, f"默认 20 轮不应降级: {demoted[:1]}"
            assert not triggered, f"默认 20 轮不应触发摘要: {triggered[:1]}"
            print("PASS 默认配置：20 轮无降级、无摘要")
            return 0

        demoted = [e for e in events if e.get("event") == "layer1 降级"]
        triggered = [e for e in events if e.get("event") == "summary trigger"]
        done = [e for e in events if e.get("event") == "summary done"]
        assert demoted, "演示配置未出现 layer1 降级"
        assert triggered, "演示配置未出现 summary trigger"
        assert done, "演示配置未出现 summary done"
        answer = await _ask(
            client, base_url, session_id, "最开始那个订单后来怎么说？"
        )
        text = answer.get("answer") or ""
        assert "1001" in text or "猫爬架" in text, f"摘要未回答问题: {text}"
        print("PASS 演示配置：降级/触发/完成级联，摘要可回答早期订单")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["default", "demo"], default="default")
    parser.add_argument("--turns", type=int, default=22)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    return asyncio.run(run(args.mode, args.turns, args.base_url))


if __name__ == "__main__":
    raise SystemExit(main())
