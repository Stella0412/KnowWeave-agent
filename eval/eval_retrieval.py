"""在线检索质量评估 + 相似度阈值标定。

前置：先跑过 `python -m app.jobs.build_kb`（必要时再跑 `mine_qa`），Milvus 里有数据。
用法：uv run python eval/eval_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import settings
from app.core.embeddings import embed_texts
from app.core.retrieval import search_knowledge
from app.kb.milvus_client import build_milvus_client


def _load(path):
    import json

    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def topic_matched(hits, topic):
    if topic is None:
        return not hits
    for h in hits:
        if topic in f"{h.question}\n{h.answer}":
            return True
    return False


def main() -> None:
    store = build_milvus_client(settings)
    store.ensure_collection()
    samples = _load("eval/samples_retrieval.jsonl")

    print(f"集合知识条数: {store.count()}\n")
    probes = []
    for s in samples:
        vector = embed_texts([s["query"]])[0]
        raw = store.search(vector, settings.retrieval_top_k)
        probes.append((s, raw))
        top = raw[0] if raw else None
        label = f"{top['question'].splitlines()[0][:20]}|{top['score']:.3f}" if top else "（无）"
        print(f"{s['query']:<26} {(top['score'] if top else 0):>10.4f}  {label}")

    hit_scores = [h[0]["score"] for s, h in probes if s["should_hit"] and h]
    miss_scores = [h[0]["score"] for s, h in probes if not s["should_hit"] and h]
    low, high = max(miss_scores, default=0.0), min(hit_scores, default=1.0)
    print(
        f"\n应命中样例的 Top1 最低相似度 = {high:.4f}"
        f"\n应拒绝样例的 Top1 最高相似度 = {low:.4f}"
        f"\n→ 阈值建议区间 ({low:.4f}, {high:.4f})，当前配置 = {settings.retrieval_min_score}"
    )

    ok = 0
    print()
    for s in samples:
        hits = search_knowledge(s["query"], settings.retrieval_top_k)
        matched = topic_matched(hits, s["expect_topic"])
        ok += int(matched)
        got = hits[0].question.splitlines()[0][:24] if hits else "（拒答/未找到）"
        print(f"[{s['query'][:16]:<16}] 期望={s['expect_topic'] or '拒绝'} 实际={got} {'✓' if matched else '✗'}")

    print(f"\n检索准确率: {ok}/{len(samples)}")
    store.close()


if __name__ == "__main__":
    main()
