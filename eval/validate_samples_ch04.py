"""连库校验 300 题评估集：章节存在、相关块非空、标准要点能在证据里找到。

用法：uv run python eval/validate_samples_ch04.py
它与 `eval/eval_retrieval_ch04.py` 共用 `resolve_relevant`，避免评估脚本和校验脚本
对「哪些 chunk 算相关」各有一套解释。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import settings
from app.kb.milvus_client import build_milvus_client
from eval.eval_retrieval_ch04 import load_samples, resolve_relevant, section_paths_of


def main() -> None:
    samples = load_samples()
    store = build_milvus_client(settings)
    store.ensure_collection()
    rows = store.query_by_expr("id > 0", limit=16384)
    by_id = {int(row["id"]): row for row in rows}
    known_paths = {row.get("section_path") for row in rows}

    problems: list[str] = []
    buckets = Counter(sample["bucket"] for sample in samples)
    for sample in samples:
        paths = section_paths_of(sample)
        if sample["bucket"] == "D_absent":
            if paths:
                problems.append(f"{sample['id']}: 拒答题不应有 ground-truth 章节")
            continue
        if not paths:
            problems.append(f"{sample['id']}: 缺少 ground-truth 章节")
            continue
        missing_paths = [path for path in paths if path not in known_paths]
        if missing_paths:
            problems.append(f"{sample['id']}: 章节不存在：{missing_paths}")
            continue
        relevant = resolve_relevant(rows, sample)
        if not relevant:
            problems.append(f"{sample['id']}: 相关块为空（must_contain 可能过严）")
            continue
        haystack = "\n".join(
            f"{by_id[cid].get('question') or ''}\n{by_id[cid].get('answer') or ''}"
            for cid in sorted(relevant)
        )
        for point in sample.get("expected_points") or []:
            if point not in haystack:
                problems.append(f"{sample['id']}: 要点不在相关块里：{point!r}")

    print(f"评估集：{len(samples)} 题；知识库：{len(rows)} chunks")
    print("分桶：", dict(sorted(buckets.items())))
    if problems:
        print(f"\n[问题 {len(problems)} 条]")
        for problem in problems:
            print(" -", problem)
        raise SystemExit(1)
    print("校验通过：章节存在、相关块非空、标准要点均能在相关块中找到")
    store.close()


if __name__ == "__main__":
    main()
