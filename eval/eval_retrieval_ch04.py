"""ch04 检索段评估：四策略 × 全题 → Recall@5 / Recall@10 / MRR@10，按桶与难度分列。

前置：先跑 `uv run python -m app.jobs.rebuild_kb --yes`（库要有新夹具的 chunk）。

用法：uv run python eval/eval_retrieval_ch04.py
产物：stdout 表 + eval/reports/ch04-retrieval-<date>.md（提交，验收① 的证据）

关键约定：
- ground-truth 按 **section_path** 解析成 chunk id 集合，再叠加 `must_contain` 收窄到具体 chunk。
  不用 chunk id：每次重建 id 都会漂，section 不会。
- `D_reject` **不参与** Recall/MRR 汇总（没有正确答案可召回，混进去会污染其余桶的均值），
  它单独一列「检索是否留空」。
- **`hybrid_rerank` 这一路走的是 `rerank_stage`，不是 `recall`**：`recall` 只召回、不重排，
  `recall("hybrid_rerank", ...)` 与 `recall("hybrid", ...)` 是同一个 `hybrid_search` 调用，
  报告第 4 列会与第 3 列逐字节相同（Step 1 的裁定）。故第 4 路 = 先按 `hybrid` 召回，
  再过 `rerank_stage`（与 `run_pipeline` 同一个函数）。
- 精排是**网络调用**：降级（上游挂 / 返回空表）时第 4 列**本来就该等于**第 3 列，
  所以脚本必须把「降级题数」报出来，并在**每一道题**都撞车时打印 `[警告] 精排未生效` ——
  否则验收① 会在一路静默降级的运行里「通过」。
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import settings
from app.kb.milvus_client import build_milvus_client
from app.rag.pipeline import rerank_stage
from app.rag.strategies import HYBRID, HYBRID_RERANK, STRATEGIES, embed_query, recall
from app.rag.rewrite import build_bm25_query, rewrite_query

SAMPLES = Path("eval/samples_ch04.jsonl")
REPORT_DIR = Path("eval/reports")
K_VALUES = (5, 10)
_MUST_CONTAIN_SCAN = 16384


def load_samples(path: Path = SAMPLES) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def section_paths_of(sample: dict) -> list[str]:
    ground_truth = sample.get("ground_truth") or {}
    if ground_truth.get("section_path"):
        return [ground_truth["section_path"]]
    return list(ground_truth.get("section_paths") or [])


def resolve_relevant(rows: list[dict], sample: dict) -> set[int]:
    """把 ground-truth 解析成相关 chunk id 集合。

    `must_contain` 是逐条的**全部**都要出现在该 chunk 正文里（AND），用来把整个 section
    收窄到真正答得上问题的那一块 —— 比如 12 行同构表里只留「退款」那一行。
    """
    wanted = set(section_paths_of(sample))
    if not wanted:
        return set()
    must = (sample.get("ground_truth") or {}).get("must_contain") or []
    relevant = set()
    for row in rows:
        if row.get("section_path") not in wanted:
            continue
        body = f"{row.get('question') or ''}\n{row.get('answer') or ''}"
        if all(token in body for token in must):
            relevant.add(int(row["id"]))
    return relevant


def score_one(retrieved_ids: list[int], relevant: set[int], k_values=K_VALUES) -> dict:
    """一个样本的召回指标。relevant 为空（D_reject）返回空 dict，不参与汇总。"""
    if not relevant:
        return {}
    metrics: dict[str, float] = {}
    for k in k_values:
        top = retrieved_ids[:k]
        metrics[f"recall@{k}"] = len(set(top) & relevant) / len(relevant)
    rank = next((i + 1 for i, cid in enumerate(retrieved_ids[:10]) if cid in relevant), None)
    metrics["mrr@10"] = 1.0 / rank if rank else 0.0
    return metrics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _table(title: str, groups: list[tuple[str, list[dict]]]) -> list[str]:
    header = "| 分组 | 题数 | " + " | ".join(
        m for m in ("Recall@5", "Recall@10", "MRR@10")
    ) + " |"
    lines = [f"### {title}", "", header, "|---" * (3 + len(K_VALUES)) + "|"]
    for name, metrics in groups:
        lines.append(
            f"| {name} | {len(metrics)} | "
            + " | ".join(
                f"{_mean([m[key] for m in metrics]):.3f}" for key in ("recall@5", "recall@10", "mrr@10")
            )
            + " |"
        )
    lines.append("")
    return lines


def main() -> None:
    samples = load_samples()
    store = build_milvus_client(settings)
    store.ensure_collection()
    rows = store.query_by_expr("id > 0", limit=_MUST_CONTAIN_SCAN)
    print(f"集合知识条数: {store.count()}（评估集 {len(samples)} 题）\n")

    per_sample: dict[str, list[tuple[dict, list[int]]]] = {s: [] for s in STRATEGIES}
    relevant_by_id: dict[str, set[int]] = {}
    reject_outcomes: dict[str, list[bool]] = {s: [] for s in STRATEGIES}
    rerank_degraded_ids: list[str] = []
    collapsed_on: list[str] = []

    for sample in samples:
        relevant_by_id[sample["id"]] = resolve_relevant(rows, sample)
        rewrite = rewrite_query(sample["query"])
        # 与 run_pipeline 同口径：精排与向量化都吃改写后的标准问法（没有才退回原话）
        eff_query = rewrite.standard_query or sample["query"]
        dense_vector = embed_query(eff_query)
        bm25_query = build_bm25_query(sample["query"], rewrite)
        for strategy in STRATEGIES:
            # 第 4 路只能先按 `hybrid` 召回：recall 不重排（见文件头约定）。
            # recall("hybrid_rerank", ...) 与 recall("hybrid", ...) 是同一个调用。
            base = HYBRID if strategy == HYBRID_RERANK else strategy
            candidates = recall(
                base, dense_vector=dense_vector, bm25_query=bm25_query,
                top_k=settings.recall_top_k, rrf_k=settings.rrf_k, milvus=store,
            )
            if strategy == HYBRID_RERANK and candidates:
                candidates, _rerank_score, degraded = rerank_stage(candidates, eff_query)
                if degraded:
                    rerank_degraded_ids.append(sample["id"])
            ids = [c.id for c in candidates]
            per_sample[strategy].append((sample, ids))
            if sample["bucket"] == "D_reject":
                # 该桶的指标是「是否正确拒答」，这里只看召回是否为空（有候选 = 没拦住）
                reject_outcomes[strategy].append(not ids)

    # 塌陷自检：第 4 列如果在**每道题**上都与第 3 列同 id 串，说明精排根本没参与。
    # （精排降级的个别题两列必然相同，那是正常的，所以只看「全部撞车」。）
    for (sample_h, ids_h), (_sample_r, ids_r) in zip(
        per_sample[HYBRID], per_sample[HYBRID_RERANK]
    ):
        assert sample_h["id"] == _sample_r["id"], "两列必须逐题对齐，否则这轮比对无意义"
        if ids_h == ids_r:
            collapsed_on.append(sample_h["id"])

    report: list[str] = [
        f"# ch04 四策略检索对比（{date.today().isoformat()}）",
        "",
        f"- 集合条数：{store.count()}；评估集：{len(samples)} 题；召回每条腿 Top-{settings.recall_top_k}，RRF k={settings.rrf_k}",
        "- ground-truth 按 section_path（+ must_contain 收窄）解析；chunk id 每次重建都会漂，不用它",
        "- `D_reject` 桶不参与 Recall/MRR 汇总（没有正确答案可召回），单独看「检索留空率」",
        f"- `hybrid_rerank` 这一路走 `rerank_stage`（与 `run_pipeline` 同一函数）；"
        f"精排降级 **{len(rerank_degraded_ids)}/{len(samples)}** 题",
        "",
    ]
    report += [
        f"- `hybrid_rerank` 与 `hybrid` 的 id 串在 {len(collapsed_on)}/{len(samples)} 题上相同"
        "（精排降级的那几题必然相同，属正常）",
        "",
    ]
    if len(collapsed_on) == len(samples):
        report += [
            "> ⚠️ **[警告] 精排未生效**：两列在**每一道题**上都完全相同 —— 第 4 列其实就是第 3 列，"
            "验收① 的「混合+重排」这一条没有被测到。先看上面那行精排降级计数，"
            "再确认 `rerank_stage` 真的被调用了。",
            "",
        ]
    report += ["## 总表（不含 D_reject）", ""]

    header = "| 策略 | 题数 | Recall@5 | Recall@10 | MRR@10 |"
    report += [header, "|---|---|---|---|---|"]
    for strategy in STRATEGIES:
        metrics = [
            score_one(ids, relevant_by_id[s["id"]])
            for s, ids in per_sample[strategy]
            if s["bucket"] != "D_reject"
        ]
        metrics = [m for m in metrics if m]
        report.append(
            f"| `{strategy}` | {len(metrics)} | "
            + " | ".join(
                f"{_mean([m[k] for m in metrics]):.3f}" for k in ("recall@5", "recall@10", "mrr@10")
            )
            + " |"
        )
    report.append("")

    for bucket in ("A_policy", "B_model", "C_colloquial", "E_multi"):
        groups = []
        for strategy in STRATEGIES:
            metrics = [
                score_one(ids, relevant_by_id[s["id"]])
                for s, ids in per_sample[strategy]
                if s["bucket"] == bucket
            ]
            groups.append((f"`{strategy}`", [m for m in metrics if m]))
        report += _table(f"分桶：{bucket}", groups)

    for difficulty in ("easy", "medium", "hard"):
        groups = []
        for strategy in STRATEGIES:
            metrics = [
                score_one(ids, relevant_by_id[s["id"]])
                for s, ids in per_sample[strategy]
                if s["difficulty"] == difficulty and s["bucket"] != "D_reject"
            ]
            groups.append((f"`{strategy}`", [m for m in metrics if m]))
        report += _table(f"分难度：{difficulty}", groups)

    report += ["### D_reject 桶：检索留空率（越高越好）", "", "| 策略 | 留空率 |", "|---|---|"]
    for strategy in STRATEGIES:
        outcomes = reject_outcomes[strategy]
        rate = _mean([1.0 if ok else 0.0 for ok in outcomes])
        report.append(f"| `{strategy}` | {rate:.3f}（{sum(outcomes)}/{len(outcomes)}） |")
    report.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"ch04-retrieval-{date.today().isoformat()}.md"
    text = "\n".join(report)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[报告已写入] {out_path}")
    store.close()


if __name__ == "__main__":
    main()
