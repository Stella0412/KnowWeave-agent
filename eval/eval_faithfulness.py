"""ch04 生成段评估：忠实度裁判 + 拒答正确率 + 负面断言检查。

前置：先跑 `uv run python -m app.jobs.rebuild_kb --yes`（库要有新夹具的 chunk）。

用法：uv run python eval/eval_faithfulness.py
产物：stdout 表 + eval/reports/ch04-faithfulness-<date>.md；
      判为编造的题写入 `faith_cases`（含证据全集快照，便于人工追溯「手里有哪几条、引了哪几条」）。

metric 说明：
- 常规题：Faithfulness 率（有依据的答案数 / 生成答案数），按桶列。
- `D_reject` 桶：**拒答正确率**（拒答没有可判的断言，忠实度对它没意义）。
- `no_commitment` 题：另跑负面断言检查 —— 库里的依据本身是「不承诺具体数字」，
  答出具体到账天数就是错，这是 §9.4 负面知识规则唯一能被机械检验的地方。
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import settings
from app.core.llm import build_chat_model, build_structured_model
from app.db.engine import SessionFactory
from app.db.repository import FaithCaseRepository
from app.prompts.customer_service import SYSTEM_PROMPT
from app.prompts.rag import FaithJudgeOutput, faith_judge_messages
from app.rag.pipeline import default_source_files, run_pipeline
from app.rag.refusal import REFUSAL_TEXT

from langchain_core.messages import HumanMessage, SystemMessage

SAMPLES = Path("eval/samples_ch04.jsonl")
REPORT_DIR = Path("eval/reports")

# 时间量词 + 到账类动词，**两者同现**才算违规（否则「3 个工作日处理」会被误伤）
_TIME_PATTERN = re.compile(r"\d+\s*(?:个)?\s*(?:工作日|天|日|小时|分钟)|(?:\d{1,2}\s*月\s*\d{1,2}\s*日)")
_ARRIVAL_PATTERN = re.compile(r"到账|到账时间|退款到|打款|退回|返回到?|入账|收到退款")


def has_commitment_number(text: str) -> bool:
    """句子里同时有具体时间量词与到账类动词 → 视为给出了到账承诺。"""
    return bool(_TIME_PATTERN.search(text) and _ARRIVAL_PATTERN.search(text))


def check_no_commitment(answer: str) -> tuple[bool, list[str]]:
    """返回 (是否合规, 违规片段)。逐句判，避免一句话里的无关时间量词污染整段。"""
    violations = []
    for sentence in re.split(r"[。！？\n]", answer or ""):
        if has_commitment_number(sentence):
            violations.append(sentence.strip())
    return (not violations), violations


def generate_answer(model, query: str, result) -> str:
    """复刻线上生成：同样的 System Prompt + 证据块，问同一句用户原话。"""
    if result.refused:
        return REFUSAL_TEXT
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"用户问题：{query}\n\n知识库证据：\n{result.evidence_text}"),
    ]
    return str(model.invoke(messages).content or "")


def judge(judge_model, query: str, answer: str, evidence_block: str) -> FaithJudgeOutput:
    return judge_model.invoke(faith_judge_messages(query, answer, evidence_block))


def main() -> None:
    with SAMPLES.open(encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]

    generator = build_chat_model(settings)
    judge_model = build_structured_model(settings, FaithJudgeOutput)
    source_files = default_source_files()
    judge_name = settings.llm_model

    per_bucket: dict[str, dict] = {}
    reject_results: list[bool] = []
    negative_checks: list[tuple[str, bool, list[str]]] = []
    fabrication_rows: list[dict] = []
    degraded_ids: list[str] = []

    for sample in samples:
        result = run_pipeline(sample["query"], source_files=source_files)
        # 降级题必须记账：精排挂了这条题是按 RRF 顺序生成的、改写挂了是按用户原话检索的。
        # 不记的话，报告会顶着「策略 hybrid_rerank」的抬头，给出一份**部分没重排**的分数。
        if result.rerank_degraded or result.rewrite_degraded:
            degraded_ids.append(sample["id"])
        bucket = sample["bucket"]
        slot = per_bucket.setdefault(bucket, {"total": 0, "faithful": 0})
        slot["total"] += 1

        if sample["expected_behavior"] == "refuse":
            reject_results.append(result.refused)
            if not result.refused:
                # 该拒没拒：把生成的答案当编造个案记下来（它一定是在没依据的情况下答了）
                fabrication_rows.append({
                    "sample": sample, "answer": generate_answer(generator, sample["query"], result),
                    "reason": "应拒答却给出了回答，且知识库里没有对应依据",
                    "citations": result.citations, "query": sample["query"],
                })
            continue

        answer = generate_answer(generator, sample["query"], result)
        verdict = judge(judge_model, sample["query"], answer, result.evidence_text)
        if verdict.faithful:
            slot["faithful"] += 1
        else:
            fabrication_rows.append({
                "sample": sample, "answer": answer,
                "reason": verdict.reason or "（裁判未给理由）",
                "citations": result.citations, "query": sample["query"],
            })

        if sample["expected_behavior"] == "no_commitment":
            ok, hits = check_no_commitment(answer)
            negative_checks.append((sample["id"], ok, hits))

    if fabrication_rows:
        import asyncio

        async def _persist() -> None:
            async with SessionFactory() as session:
                repo = FaithCaseRepository(session)
                for row in fabrication_rows:
                    await repo.upsert_case(
                        eval_id=row["sample"]["id"], bucket=row["sample"]["bucket"],
                        query=row["query"], strategy=settings.rag_strategy,
                        answer=row["answer"], reason=row["reason"],
                        citations=row["citations"], judge_model=judge_name,
                    )
                await session.commit()

        asyncio.run(_persist())

    lines = [
        f"# ch04 生成段评估（{date.today().isoformat()}）",
        "",
        f"- 策略：`{settings.rag_strategy}`；裁判模型：`{judge_name}`；评估集：{len(samples)} 题",
        f"- **降级题：{len(degraded_ids)}/{len(samples)}**"
        + (f"（{'、'.join(degraded_ids)}）" if degraded_ids else "")
        + " —— 降级题的精排/改写没跑成，它们的 Faithfulness 不代表本策略的真实水平；"
        "降级数不为 0 时，先查上游再解读下面的数字",
        "- 判为编造的个案已写入 `faith_cases`（一题一行，含证据全集快照）",
        "",
        "## Faithfulness 率（按桶）",
        "",
        "| 桶 | 生成答案数 | 忠实数 | Faithfulness |",
        "|---|---|---|---|",
    ]
    generated_total = sum(v["total"] for k, v in per_bucket.items() if k != "D_reject")
    faithful_total = sum(v["faithful"] for k, v in per_bucket.items() if k != "D_reject")
    for bucket, slot in sorted(per_bucket.items()):
        if bucket == "D_reject":
            continue
        rate = slot["faithful"] / slot["total"] if slot["total"] else 0.0
        lines.append(f"| {bucket} | {slot['total']} | {slot['faithful']} | {rate:.3f} |")
    overall = faithful_total / generated_total if generated_total else 0.0
    lines += ["", f"**合计 Faithfulness = {overall:.3f}**（{faithful_total}/{generated_total}）", ""]

    lines += [
        "## 拒答正确率（D_reject 桶）",
        "",
        f"- 正确拒答 {sum(reject_results)}/{len(reject_results)} = "
        f"{(sum(reject_results) / len(reject_results)) if reject_results else 0:.3f}",
        "",
        "## 负面断言检查（no_commitment 题）",
        "",
    ]
    for sample_id, ok, hits in negative_checks:
        lines.append(f"- {sample_id}: {'通过' if ok else '不合规 → ' + ' / '.join(hits)}")
    lines.append("")

    lines += ["## 编造个案（已入台账）", ""]
    if fabrication_rows:
        for row in fabrication_rows:
            lines.append(f"- {row['sample']['id']}（{row['sample']['bucket']}）：{row['reason']}")
    else:
        lines.append("- 无")
    lines.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"ch04-faithfulness-{date.today().isoformat()}.md"
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[报告已写入] {out_path}")


if __name__ == "__main__":
    main()
