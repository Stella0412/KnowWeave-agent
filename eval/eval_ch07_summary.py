# eval/eval_ch07_summary.py
"""摘要 Prompt 标注评估：真实模型逐条跑，不放进 pytest。"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import settings
from app.core.llm import build_structured_model
from app.prompts.context import SummaryOutput, summary_messages
from app.services.token_counter import TokenCounter, text_of

CHITCHAT = ("谢谢", "您好", "不客气", "很高兴为您服务", "祝您生活愉快")


def _transcript(dialogue, counter):
    lines = []
    for item in dialogue:
        role = item["role"]
        content = item["content"]
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "tool":
            lines.append("工具：" + counter.truncate_text(content, 200, "…（截断）"))
        else:
            lines.append("客服：" + counter.truncate_text(content, 400, "…（截断）"))
    return "\n".join(lines)


def _score(sample, summary):
    missing = [fact for fact in sample["expected_facts"] if fact not in summary]
    invented = [fact for fact in sample["forbidden"] if fact in summary]
    length_ok = sample["min_chars"] <= len(summary) <= sample["max_chars"]
    chitchat = [word for word in CHITCHAT if word in summary]
    if sample["expected_facts"]:
        length_ok = length_ok and bool(summary)
    return {
        "recall": not missing,
        "no_invention": not invented,
        "length": length_ok,
        "no_chitchat": not chitchat,
        "missing": missing,
        "invented": invented,
        "chitchat": chitchat,
    }


async def main():
    counter = TokenCounter(
        cjk_tokens_per_char=settings.cjk_tokens_per_char,
        estimate_factor=settings.token_estimate_factor,
    )
    model = build_structured_model(
        settings, SummaryOutput, temperature=0.0,
        max_tokens=settings.summary_max_output_tokens,
    )
    samples = [
        json.loads(line)
        for line in Path("eval/samples_ch07_summary.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = 0
    for sample in samples:
        transcript = _transcript(sample["dialogue"], counter)
        result = await model.ainvoke(summary_messages(transcript))
        summary = (getattr(result, "summary", "") or "").strip()
        score = _score(sample, summary)
        ok = all(score[key] for key in ("recall", "no_invention", "length", "no_chitchat"))
        passed += int(ok)
        print(f"[{sample['id']}] {'PASS' if ok else 'FAIL'} {summary}")
        if not ok:
            print(f"    {score}")
    print(f"\n摘要评估: {passed}/{len(samples)}")
    return 0 if passed == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
