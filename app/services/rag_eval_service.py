"""RAG 评估面板的运行器。

页面上要看到的四块：
1. **四策略检索排序**：每题每策略的前 10 条候选（含命中的 section_path）；
2. **证据覆盖**：ground-truth 的 `must_contain` 关键事实是否出现在前 10 条证据里；
3. **端到端答案覆盖**：生产策略 `hybrid_rerank` 实际生成的答案是否覆盖期望行为
   （answer=关键事实、no_commitment=不承诺具体到账时间、refuse=正确拒答）；
4. **可重跑**：后台线程跑评估，页面轮询进度。

指标口径复用 `eval/` 里的函数，避免面板和 CLI 报告各算一套。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
import uuid
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.llm import build_chat_model, build_structured_model
from app.kb.milvus_client import get_milvus_client
from app.prompts.rag import FaithJudgeOutput
from app.rag.evidence import Evidence
from app.rag.pipeline import default_source_files, rerank_stage, run_pipeline
from app.rag.refusal import refusal_text
from app.rag.strategies import HYBRID, HYBRID_RERANK, STRATEGIES, embed_query, recall
from app.rag.rewrite import build_bm25_query, rewrite_query
from eval.eval_faithfulness import check_no_commitment, generate_answer, judge
from eval.eval_retrieval_ch04 import (
    load_samples,
    resolve_relevant,
    score_one,
    section_paths_of,
)

logger = logging.getLogger(__name__)

RUN_DIR = Path("data/rag_eval")
LATEST_PATH = RUN_DIR / "latest.json"
MUST_CONTAIN_SCAN = 16384
VALID_MODES = ("retrieval", "full")

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "status": "idle",
    "run_id": None,
    "mode": None,
    "progress": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _models_snapshot() -> dict:
    """本轮评估实际用的模型，页头直接展示；写进结果快照，之后改配置也不会漂。"""
    return {
        "embed": settings.embed_model,
        "rerank": settings.rerank_model,
        "judge": settings.llm_model,
    }


def expected_tokens(sample: dict) -> list[str]:
    """答案/证据覆盖用的标准要点：优先 `expected_points`，旧样本回退 `must_contain`。"""
    points = sample.get("expected_points")
    if points:
        return list(points)
    return (sample.get("ground_truth") or {}).get("must_contain") or []


_CITATION = re.compile(r"\[\s*\d+\s*\]")
_NON_CONTENT = re.compile(r"[^0-9a-z一-鿿]+")


def _coverage_key(text: str | None) -> str:
    """覆盖比对用的规范形：全角转半角、小写，去掉引用编号、空白、标点与 markdown 强调符。

    答案里的 `**5000 元**起`、`[1][10]` 这类排版不该影响「要点有没有答到」。
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    return _NON_CONTENT.sub("", _CITATION.sub("", text))


def _covers(point: str, haystack: str) -> bool:
    """要点里的字是否都出现在文本里（顺序不限）。

    机械近似而非语义判断：同义改写（「另计」写成「加收 12 元」）仍会漏计，
    宁可漏计也不让排版差异把答对的题判错。
    """
    if not point:
        return False
    return not (Counter(point) - Counter(haystack))


def token_coverage(tokens: list[str] | None, text: str | None) -> dict:
    """期望关键事实在文本里出现了几条。去重但保序，空 token 列表不算分。"""
    wanted = list(dict.fromkeys(t for t in (tokens or []) if t))
    if not wanted:
        return {"matched": [], "missing": [], "rate": None}
    haystack = _coverage_key(text)
    matched = [token for token in wanted if _covers(_coverage_key(token), haystack)]
    missing = [token for token in wanted if token not in matched]
    return {"matched": matched, "missing": missing, "rate": len(matched) / len(wanted)}


def first_relevant_rank(ids: list[int], relevant: set[int]) -> int | None:
    return next((index + 1 for index, cid in enumerate(ids) if cid in relevant), None)


def evidence_coverage(
    hits: list[dict], must_contain: list[str] | None, relevant_ids: set[int]
) -> dict:
    """前 10 条证据对 ground-truth 的覆盖。

    有标准要点（`expected_points`，旧样本回退 `must_contain`）时按关键事实算；
    没有时退化为「前 10 条里有没有命中相关 chunk」。
    `D_absent` 没有真值，返回 rate=None，汇总时排除。
    """
    text = "\n".join(
        f"{hit.get('question') or ''}\n{hit.get('answer') or ''}" for hit in (hits or [])
    )
    coverage = token_coverage(must_contain, text)
    if coverage["rate"] is not None:
        return {**coverage, "basis": "must_contain"}
    if not relevant_ids:
        return {**coverage, "basis": "no_ground_truth"}
    hit_any = any(int(hit.get("id", -1)) in relevant_ids for hit in (hits or []))
    return {
        **coverage,
        "rate": 1.0 if hit_any else 0.0,
        "basis": "relevant_chunk",
    }


def answer_coverage(sample: dict, answer: str, *, refused: bool) -> dict:
    """端到端答案是否满足该题的期望行为。"""
    behavior = sample.get("expected_behavior") or "answer"
    tokens = expected_tokens(sample)

    if behavior == "refuse":
        return {
            "ok": bool(refused),
            "rate": 1.0 if refused else 0.0,
            "basis": "refusal",
            "matched": [],
            "missing": [],
            "violations": [],
        }

    coverage = token_coverage(tokens, answer)
    if behavior == "no_commitment":
        ok, violations = check_no_commitment(answer)
        points_ok = coverage["rate"] in (None, 1.0)
        ok = bool(ok) and points_ok and not refused
        return {
            "ok": ok,
            "rate": 1.0 if ok else 0.0,
            "basis": "negative_assertion",
            "matched": coverage["matched"],
            "missing": coverage["missing"],
            "violations": violations,
        }

    if coverage["rate"] is None:
        ok = bool((answer or "").strip()) and not refused
        return {
            "ok": ok,
            "rate": 1.0 if ok else 0.0,
            "basis": "non_empty_answer",
            "matched": [],
            "missing": [],
            "violations": [],
        }
    return {
        "ok": (coverage["rate"] == 1.0) and not refused,
        "rate": 0.0 if refused else coverage["rate"],
        "basis": "must_contain",
        "matched": coverage["matched"],
        "missing": coverage["missing"],
        "violations": [],
    }


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "…"


def _hit_payload(item: Evidence, relevant: set[int]) -> dict:
    return {
        "id": item.id,
        "section_path": item.section_path,
        "question": _clip(item.question, 140),
        "answer": _clip(item.answer, 320),
        "relevant": item.id in relevant,
        "score": round(float(item.score), 6),
    }


def _strategy_summary(questions: list[dict]) -> dict:
    answerable = [q for q in questions if q.get("bucket") != "D_absent"]
    summary = {}
    for strategy in STRATEGIES:
        metrics = [
            q["strategies"][strategy]["metrics"]
            for q in answerable
            if q["strategies"][strategy].get("metrics")
        ]
        coverages = [
            q["strategies"][strategy]["evidence_coverage"]["rate"]
            for q in answerable
            if q["strategies"][strategy]["evidence_coverage"]["rate"] is not None
        ]
        summary[strategy] = {
            "questions": len(metrics),
            "recall@5": _mean([m.get("recall@5", 0.0) for m in metrics]),
            "recall@10": _mean([m.get("recall@10", 0.0) for m in metrics]),
            "mrr@10": _mean([m.get("mrr@10", 0.0) for m in metrics]),
            "evidence_coverage": _mean(coverages),
        }
    return summary


def _answer_summary(questions: list[dict]) -> dict | None:
    answered = [q for q in questions if q.get("answer_coverage")]
    if not answered:
        return None
    covered = [q for q in answered if q["answer_coverage"]["ok"]]
    refuse_rows = [q for q in answered if q.get("bucket") == "D_absent"]
    answerable_rows = [q for q in answered if q.get("bucket") != "D_absent"]
    faithful_rows = [q for q in answered if q.get("faithful") is not None]
    negative_rows = [q for q in answered if q.get("negative") is not None]
    return {
        "answered": len(answered),
        "covered": len(covered),
        "coverage": len(covered) / len(answered),
        "refusal_correct": sum(1 for q in refuse_rows if q.get("refused")),
        "refusal_total": len(refuse_rows),
        "refused_answerable": sum(1 for q in answerable_rows if q.get("refused")),
        "answerable_total": len(answerable_rows),
        "faithful": sum(1 for q in faithful_rows if q.get("faithful")),
        "faithful_total": len(faithful_rows),
        "negative_ok": sum(1 for q in negative_rows if q["negative"]["ok"]),
        "negative_total": len(negative_rows),
    }


def _set_progress(run_id: str, phase: str, done: int, total: int, message: str) -> None:
    with _state_lock:
        if _state.get("run_id") != run_id:
            return
        _state["progress"] = {
            "phase": phase,
            "done": done,
            "total": total,
            "message": message,
        }


def _run_retrieval_pass(
    run_id: str, samples: list[dict], rows: list[dict], store, done: int, total: int
) -> tuple[list[dict], int]:
    questions = []
    relevant_by_id = {sample["id"]: resolve_relevant(rows, sample) for sample in samples}
    for sample in samples:
        _set_progress(run_id, "retrieval", done, total, f"检索 {sample['id']}…")
        rewrite = rewrite_query(sample["query"])
        effective_query = rewrite.standard_query or sample["query"]
        dense_vector = embed_query(effective_query)
        bm25_query = build_bm25_query(sample["query"], rewrite)
        relevant = relevant_by_id[sample["id"]]
        strategies = {}
        for strategy in STRATEGIES:
            base = HYBRID if strategy == HYBRID_RERANK else strategy
            candidates = recall(
                base,
                dense_vector=dense_vector,
                bm25_query=bm25_query,
                top_k=settings.recall_top_k,
                rrf_k=settings.rrf_k,
                milvus=store,
            )
            degraded = False
            if strategy == HYBRID_RERANK and candidates:
                candidates, _rerank_score, degraded = rerank_stage(candidates, effective_query)
            ids = [item.id for item in candidates]
            top_hits = [
                {"id": item.id, "question": item.question, "answer": item.answer}
                for item in candidates[:10]
            ]
            strategies[strategy] = {
                "ids": ids,
                "hits": [_hit_payload(item, relevant) for item in candidates[:10]],
                "metrics": score_one(ids, relevant),
                "first_relevant_rank": first_relevant_rank(ids, relevant),
                "evidence_coverage": evidence_coverage(
                    top_hits,
                    expected_tokens(sample),
                    relevant,
                ),
                "degraded": degraded,
            }
        questions.append({
            "id": sample["id"],
            "bucket": sample.get("bucket"),
            "difficulty": sample.get("difficulty"),
            "expected_behavior": sample.get("expected_behavior"),
            "query": sample["query"],
            "notes": sample.get("notes") or "",
            "ground_truth": sample.get("ground_truth") or {},
            "section_paths": section_paths_of(sample),
            "relevant_ids": sorted(relevant),
            "strategies": strategies,
        })
        done += 1
    return questions, done


def _run_answer_pass(
    run_id: str, samples: list[dict], questions: list[dict], done: int, total: int
) -> int:
    by_id = {q["id"]: q for q in questions}
    generator = build_chat_model(settings)
    judge_model = build_structured_model(settings, FaithJudgeOutput)
    source_files = default_source_files()

    for sample in samples:
        _set_progress(run_id, "answer", done, total, f"生成 {sample['id']}…")
        question = by_id[sample["id"]]
        result = run_pipeline(sample["query"], source_files=source_files)
        question["refused"] = bool(result.refused)
        question["refusal_source"] = result.refusal_source
        question["refusal_reason"] = result.refusal_reason
        question["degraded"] = bool(result.rerank_degraded or result.rewrite_degraded)
        question["citations"] = result.citations

        if result.refused:
            answer = refusal_text(sample["query"])
            question["answer"] = answer
            question["answer_coverage"] = answer_coverage(sample, answer, refused=True)
            question["faithful"] = None
            question["judge_reason"] = None
        else:
            answer = generate_answer(generator, sample["query"], result)
            question["answer"] = answer
            question["answer_coverage"] = answer_coverage(sample, answer, refused=False)
            if sample.get("expected_behavior") == "refuse":
                question["faithful"] = None
                question["judge_reason"] = None
            else:
                verdict = judge(judge_model, sample["query"], answer, result.evidence_text)
                question["faithful"] = bool(verdict.faithful)
                question["judge_reason"] = verdict.reason or ""

        if sample.get("expected_behavior") == "no_commitment":
            ok, violations = check_no_commitment(answer)
            question["negative"] = {"ok": bool(ok), "violations": violations}
        done += 1
    return done


def run_evaluation(mode: str = "retrieval", run_id: str = "manual") -> dict:
    """同步跑一轮评估，返回可序列化结果；由后台线程调用。"""
    if mode not in VALID_MODES:
        raise ValueError(f"未知评估模式：{mode}（可选：{'/'.join(VALID_MODES)}）")

    samples = load_samples()
    store = get_milvus_client()
    store.ensure_collection()
    rows = store.query_by_expr("id > 0", limit=MUST_CONTAIN_SCAN)
    collection_count = store.count()
    total = len(samples) * (2 if mode == "full" else 1)

    started_at = _now()
    questions, done = _run_retrieval_pass(run_id, samples, rows, store, 0, total)
    if mode == "full":
        done = _run_answer_pass(run_id, samples, questions, done, total)

    result = {
        "run_id": run_id,
        "mode": mode,
        "started_at": started_at,
        "finished_at": _now(),
        "collection_count": collection_count,
        "samples": len(samples),
        "models": _models_snapshot(),
        "degraded": sum(1 for q in questions if q.get("degraded")),
        "strategies": _strategy_summary(questions),
        "answers": _answer_summary(questions),
        "questions": questions,
    }
    _set_progress(run_id, "done", total, total, "完成")
    return result


def persist_result(result: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LATEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    tmp.replace(LATEST_PATH)


def read_latest() -> dict | None:
    if not LATEST_PATH.exists():
        return None
    result = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
    # 旧结果文件没有模型快照：用当前配置兜底，页头不至于空着；新结果以快照为准。
    result.setdefault("models", _models_snapshot())
    return result


def _run_guarded(run_id: str, mode: str) -> None:
    try:
        result = run_evaluation(mode, run_id)
        persist_result(result)
        with _state_lock:
            _state.update(status="done", finished_at=_now(), error=None)
    except Exception as exc:  # noqa: BLE001 —— 后台线程必须把错误交回页面
        logger.exception("RAG 评估运行失败")
        with _state_lock:
            _state.update(status="error", finished_at=_now(), error=str(exc))


def start_run(mode: str) -> tuple[bool, dict]:
    if mode not in VALID_MODES:
        raise ValueError(f"未知评估模式：{mode}（可选：{'/'.join(VALID_MODES)}）")
    with _state_lock:
        if _state["status"] == "running":
            return False, deepcopy(_state)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        _state.update({
            "status": "running",
            "run_id": run_id,
            "mode": mode,
            "progress": {"phase": "starting", "done": 0, "total": None, "message": "准备中…"},
            "error": None,
            "started_at": _now(),
            "finished_at": None,
        })
        thread = threading.Thread(
            target=_run_guarded, args=(run_id, mode), daemon=True, name=f"rag-eval-{run_id}"
        )
        thread.start()
        return True, deepcopy(_state)


def get_status() -> dict:
    with _state_lock:
        return deepcopy(_state)
