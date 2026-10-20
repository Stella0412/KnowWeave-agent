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
from app.db.engine import SessionFactory
from app.db.models import FaithCaseStatus
from app.db.repository import FaithCaseRepository
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
FAITH_SYNC_PATH = RUN_DIR / "faith_synced.json"
# 断点续跑：300 题 full 要跑几十分钟，中途断网不该把已完成的部分作废。
CHECKPOINT_PATH = RUN_DIR / "checkpoint.json"
MUST_CONTAIN_SCAN = 16384
VALID_MODES = ("retrieval", "full")
FAITH_STATUSES = tuple(status.value for status in FaithCaseStatus)

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


def _save_checkpoint(meta: dict, questions: list[dict]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "saved_at": _now(), "questions": questions}
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def _load_checkpoint(mode: str, sample_ids: list[str]) -> dict | None:
    """只认同一模式、同一批题目的检查点；评估集变了就重新开始，不混用旧结果。"""
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("mode") != mode or payload.get("sample_ids") != sample_ids:
        return None
    if not isinstance(payload.get("questions"), list):
        return None
    return payload


def _clear_checkpoint() -> None:
    CHECKPOINT_PATH.unlink(missing_ok=True)


def _run_retrieval_pass(
    run_id: str,
    samples: list[dict],
    rows: list[dict],
    store,
    done: int,
    total: int,
    *,
    existing: list[dict] | None = None,
    checkpoint_meta: dict | None = None,
) -> tuple[list[dict], int]:
    questions = list(existing or [])
    finished = {question.get("id") for question in questions}
    relevant_by_id = {
        sample["id"]: resolve_relevant(rows, sample)
        for sample in samples
        if sample["id"] not in finished
    }
    for sample in samples:
        if sample["id"] in finished:
            done += 1
            continue
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
        if checkpoint_meta and len(questions) % 10 == 0:
            _save_checkpoint(checkpoint_meta, questions)
    if checkpoint_meta:
        _save_checkpoint(checkpoint_meta, questions)
    return questions, done


def _run_answer_pass(
    run_id: str,
    samples: list[dict],
    questions: list[dict],
    done: int,
    total: int,
    *,
    checkpoint_meta: dict | None = None,
) -> int:
    by_id = {q["id"]: q for q in questions}
    generator = build_chat_model(settings)
    judge_model = build_structured_model(settings, FaithJudgeOutput, temperature=0.0)
    source_files = default_source_files()

    for sample in samples:
        question = by_id[sample["id"]]
        if "answer" in question:
            done += 1
            continue
        _set_progress(run_id, "answer", done, total, f"生成 {sample['id']}…")
        result = run_pipeline(sample["query"], source_files=source_files)
        question["refused"] = bool(result.refused)
        question["refusal_source"] = result.refusal_source
        question["refusal_reason"] = result.refusal_reason
        question["degraded"] = bool(result.rerank_degraded or result.rewrite_degraded)
        question["citations"] = result.citations
        # 低置信拒答的回溯依据：这轮实际改写成了什么、精排给了多少分。
        # 检索段与答案段各改写一次，分开记才能判断低分该怪改写还是怪精排。
        question["answer_standard_query"] = result.standard_query
        question["rerank_score"] = result.rerank_score

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
        if checkpoint_meta and done % 5 == 0:
            _save_checkpoint(checkpoint_meta, questions)
    if checkpoint_meta:
        _save_checkpoint(checkpoint_meta, questions)
    return done


def run_evaluation(mode: str = "retrieval", run_id: str = "manual") -> dict:
    """同步跑一轮评估，返回可序列化结果；由后台线程调用。"""
    if mode not in VALID_MODES:
        raise ValueError(f"未知评估模式：{mode}（可选：{'/'.join(VALID_MODES)}）")

    samples = load_samples()
    sample_ids = [sample["id"] for sample in samples]
    checkpoint = _load_checkpoint(mode, sample_ids)
    store = get_milvus_client()
    store.ensure_collection()
    rows = store.query_by_expr("id > 0", limit=MUST_CONTAIN_SCAN)
    collection_count = store.count()
    total = len(samples) * (2 if mode == "full" else 1)

    # 续跑时沿用第一次启动的时间，结论文档里看到的是整轮耗时。
    started_at = (checkpoint or {}).get("started_at") or _now()
    checkpoint_meta = {
        "mode": mode,
        "sample_ids": sample_ids,
        "started_at": started_at,
    }
    questions = (checkpoint or {}).get("questions") or []
    questions, done = _run_retrieval_pass(
        run_id,
        samples,
        rows,
        store,
        0,
        total,
        existing=questions,
        checkpoint_meta=checkpoint_meta,
    )
    if mode == "full":
        done = _run_answer_pass(
            run_id,
            samples,
            questions,
            done,
            total,
            checkpoint_meta=checkpoint_meta,
        )

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
    # 本轮已经完整跑完，删掉检查点，下一次「重跑」从零开始。
    _clear_checkpoint()
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


# ---------------------------------------------------------------------------
# 编造个案台账：本轮判出 + 人工处置状态
# ---------------------------------------------------------------------------


def _current_faith_cases(latest: dict | None) -> list[dict]:
    """从最近一轮 full 评估里取出「裁判判编造 + 该拒没拒」两类个案。

    只算本轮：判出率的分母是评估题数，分子是这两类；历史题在台账里跨轮累计，
    但不参与本轮指标，否则一轮旧账会把确认幻觉率一直拖住。
    """
    if not latest or not latest.get("answers"):
        return []
    judge_model = (latest.get("models") or {}).get("judge") or settings.llm_model
    cases: list[dict] = []
    for question in latest.get("questions", []):
        base = {
            "eval_id": question.get("id"),
            "bucket": question.get("bucket") or "unknown",
            "query": question.get("query") or "",
            "answer": question.get("answer") or "",
            "citations": question.get("citations") or [],
            "judge_model": judge_model,
        }
        if question.get("faithful") is False:
            cases.append({
                **base,
                "source": "judge",
                "reason": question.get("judge_reason") or "（裁判未给理由）",
            })
        elif (
            question.get("expected_behavior") == "refuse"
            and question.get("refused") is False
        ):
            cases.append({
                **base,
                "source": "missed_refusal",
                "reason": "应拒答却给出了回答，且知识库里没有对应依据",
            })
    return cases


def _read_faith_sync_marker() -> dict:
    if not FAITH_SYNC_PATH.exists():
        return {}
    try:
        return json.loads(FAITH_SYNC_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_faith_sync_marker(run_id: str, count: int) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "synced_at": _now(), "count": count}
    FAITH_SYNC_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


async def sync_faith_cases(force: bool = False) -> int:
    """把本轮判出的个案写进台账；同一 run_id 只写一次，避免刷页面把处置状态冲掉。"""
    latest = read_latest()
    if not latest:
        return 0
    run_id = latest.get("run_id")
    marker = _read_faith_sync_marker()
    if not force and run_id and marker.get("run_id") == run_id:
        return 0

    cases = _current_faith_cases(latest)
    if cases:
        async with SessionFactory() as session:
            repo = FaithCaseRepository(session)
            for case in cases:
                await repo.upsert_case(
                    eval_id=case["eval_id"],
                    bucket=case["bucket"],
                    query=case["query"],
                    strategy=getattr(settings, "rag_strategy", "hybrid_rerank"),
                    answer=case["answer"],
                    reason=case["reason"],
                    citations=case["citations"],
                    judge_model=case["judge_model"],
                )
            await session.commit()
    if run_id:
        _write_faith_sync_marker(run_id, len(cases))
    return len(cases)


def _faith_row_payload(row, fallback: dict) -> dict:
    status = row.status.value if hasattr(row.status, "value") else str(row.status)
    return {
        **fallback,
        "status": status,
        "seen_count": row.seen_count,
        "first_seen_at": row.first_seen_at.isoformat(timespec="seconds")
        if row.first_seen_at
        else None,
        "last_seen_at": row.last_seen_at.isoformat(timespec="seconds")
        if row.last_seen_at
        else None,
        "resolution": row.resolution,
        "resolved_at": row.resolved_at.isoformat(timespec="seconds")
        if row.resolved_at
        else None,
    }


async def list_faith_cases() -> dict:
    """台账页数据：本轮指标 + 本轮个案 + 跨轮累计。"""
    await sync_faith_cases()
    latest = read_latest()
    current = _current_faith_cases(latest)
    async with SessionFactory() as session:
        repo = FaithCaseRepository(session)
        rows = await repo.get_cases([case["eval_id"] for case in current])
        counts = await repo.count_by_status()

    payloads = []
    for case in current:
        row = rows.get(case["eval_id"])
        if row is None:
            payloads.append({
                **case,
                "status": FaithCaseStatus.OPEN.value,
                "seen_count": 1,
                "first_seen_at": None,
                "last_seen_at": None,
                "resolution": None,
                "resolved_at": None,
            })
        else:
            payloads.append(_faith_row_payload(row, case))

    status_order = {
        FaithCaseStatus.OPEN.value: 0,
        FaithCaseStatus.RESOLVED.value: 1,
        FaithCaseStatus.NOT_NEEDED.value: 2,
    }
    payloads.sort(key=lambda item: (status_order.get(item["status"], 9), item["eval_id"]))

    samples = int((latest or {}).get("samples") or len(payloads) or 0)
    judged = sum(1 for item in payloads if item["source"] == "judge")
    missed = sum(1 for item in payloads if item["source"] == "missed_refusal")
    resolved = sum(
        1 for item in payloads if item["status"] == FaithCaseStatus.RESOLVED.value
    )
    not_needed = sum(
        1 for item in payloads if item["status"] == FaithCaseStatus.NOT_NEEDED.value
    )
    open_cases = sum(
        1 for item in payloads if item["status"] == FaithCaseStatus.OPEN.value
    )
    denom = samples or 1

    return {
        "run_id": (latest or {}).get("run_id"),
        "samples": samples,
        "round": {
            "judged": judged,
            "missed_refusals": missed,
            "resolved": resolved,
            "not_needed": not_needed,
            "open": open_cases,
            "judged_rate": (judged + missed) / denom,
            "confirmed_rate": resolved / denom,
        },
        "ledger": {
            "total": sum(counts.values()),
            "open": counts.get(FaithCaseStatus.OPEN.value, 0),
            "resolved": counts.get(FaithCaseStatus.RESOLVED.value, 0),
            "not_needed": counts.get(FaithCaseStatus.NOT_NEEDED.value, 0),
        },
        "cases": payloads,
    }


async def resolve_faith_case(
    eval_id: str, status: str, resolution: str | None = None
) -> dict | None:
    """人工处置：标已解决/无需解决必须留一句交代；退回未解决时清空交代。"""
    status = (status or "").strip()
    if status not in FAITH_STATUSES:
        raise ValueError(f"未知处置状态：{status}（可选：{'/'.join(FAITH_STATUSES)}）")
    resolution = (resolution or "").strip()
    if status != FaithCaseStatus.OPEN.value and not resolution:
        raise ValueError("处置必须留一句交代：怎么解决的，或为什么不用改")
    if len(resolution) > 300:
        raise ValueError("处置说明最多 300 字")

    await sync_faith_cases()
    async with SessionFactory() as session:
        repo = FaithCaseRepository(session)
        row = await repo.set_status(eval_id, status, resolution or None)
        if row is None:
            return None
        await session.commit()
        fallback = {
            "eval_id": row.eval_id,
            "bucket": row.bucket,
            "query": row.query,
            "answer": row.answer,
            "reason": row.reason,
            "citations": row.citations or [],
            "source": "judge",
            "judge_model": row.judge_model,
        }
        return _faith_row_payload(row, fallback)
