"""挖知识抽取质量评估（标注样例 eval，替代 TDD）。

用法：uv run python eval/eval_mining.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.core.llm import build_structured_model
from app.kb.dedup import qa_fingerprint, normalize_text
from app.kb.mining import MINING_SYSTEM_PROMPT, QABatch


def expect_matched(expect: dict, got) -> bool:
    """期望问答是否被抽出：归一化后**包含**即算命中。

    不用 `qa_fingerprint` 逐字相等 — 那衡量的是「措辞有没有复刻」：LLM 抽的是原话
    （「你们发什么快递呀」「默认中通快递，偏远地区可能转其他快递。」），与标注里的
    精简版永远不等，精确率/召回率会恒为 0。这里衡量的是「事实有没有抽到」。
    """
    eq, ea = normalize_text(expect["question"]), normalize_text(expect["answer"])
    return any(eq in normalize_text(p.question) and ea in normalize_text(p.answer) for p in got)


async def main() -> None:
    model = build_structured_model(settings, QABatch)
    with open("eval/samples_mining.jsonl", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    total_expected = total_got = matched = 0
    for s in samples:
        messages = [
            SystemMessage(content=MINING_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(s["conversation"])),
        ]
        result = await model.ainvoke(messages)
        got = result.pairs if isinstance(result, QABatch) else QABatch.model_validate(result).pairs
        expect_fps = {qa_fingerprint(p["question"], p["answer"]) for p in s["expect"]}
        got_fps = {qa_fingerprint(p.question, p.answer) for p in got}
        hits = sum(1 for p in s["expect"] if expect_matched(p, got))
        total_expected += len(expect_fps)
        total_got += len(got_fps)
        matched += hits
        print(
            f"[{s['conversation'][0][:16]}…] 期望 {len(expect_fps)} 条 / "
            f"抽出 {len(got_fps)} 条 / 命中 {hits}"
        )
        for p in got:
            print(f"    · 抽到: 问={p.question!r} 答={p.answer!r}")

    precision = matched / total_got if total_got else 0.0
    recall = matched / total_expected if total_expected else 0.0
    print(f"\n抽取精确率={precision:.2f} 召回率={recall:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
