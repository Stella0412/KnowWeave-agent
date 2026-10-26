from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Protocol

import httpx

from app.config import settings
from app.db.engine import SessionFactory, engine
from app.db.repository import TopicRepository

_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "config" / "ch10_taxonomy.json"
_KNOWN_LABELS = frozenset(
    item["name"]
    for item in json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))["labels"]
)


class TopicClient(Protocol):
    async def classify(self, texts: list[str]) -> dict: ...


class HttpTopicClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def classify(self, texts: list[str]) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/classify", json={"texts": texts})
            response.raise_for_status()
            return response.json()


async def run_classify(session_factory, client, *, batch_size=20, limit=None,
                       dry_run=False, reclassify_all=False, question_ids=None,
                       exclude_patterns=()) -> dict:
    async with session_factory() as session:
        repo = TopicRepository(session)
        if question_ids:
            roots = await repo.list_roots_by_ids(list(question_ids))
        elif reclassify_all:
            from sqlalchemy import select
            from app.db.models import LowConfidenceQuestion
            roots = list((await session.execute(
                select(LowConfidenceQuestion)
                .where(LowConfidenceQuestion.merged_into_id.is_(None))
                .order_by(LowConfidenceQuestion.id)
            )).scalars())
        else:
            roots = await repo.list_unclassified_roots()
        roots = [row for row in roots if not any(p in row.raw_question for p in exclude_patterns)]
        if limit:
            roots = roots[:limit]
        explicit = bool(question_ids or reclassify_all or limit)
        if not explicit:
            roots = roots[:batch_size]
        if dry_run:
            return {"selected": len(roots), "classified": 0, "skipped": False,
                    "dry_run": True, "questions": [row.raw_question for row in roots]}
        if not roots or (not explicit and len(roots) < batch_size):
            return {"selected": len(roots), "classified": 0, "skipped": True,
                    "dry_run": False, "batch_size": batch_size}
        response = await client.classify([row.raw_question for row in roots])
        results = response.get("results") or []
        if len(results) != len(roots):
            raise RuntimeError(f"服务返回条数不匹配: {len(results)} != {len(roots)}")
        pairs = []
        counts = Counter()
        for row, items in zip(roots, results):
            labels = [item["label"] for item in items]
            if not labels:
                raise ValueError(f"问题 {row.id} 没有返回任何标签")
            unknown = [label for label in labels if label not in _KNOWN_LABELS]
            if unknown:
                raise ValueError(f"问题 {row.id} 返回了 17 类之外的标签: {unknown}")
            pairs.append((row.id, labels))
            counts.update(labels)
        await repo.upsert_many(pairs)
        await session.commit()
        return {
            "selected": len(roots), "classified": len(pairs), "skipped": False,
            "dry_run": False, "model_version": response.get("model_version"),
            "labels": dict(counts),
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ch10 旁路主题批量归类")
    parser.add_argument("--service-url", default=settings.topic_service_url)
    parser.add_argument("--batch-size", type=int, default=settings.topic_batch_size)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reclassify-all", action="store_true")
    parser.add_argument("--question-id", type=int, action="append")
    parser.add_argument("--exclude-pattern", action="append", default=[])
    return parser.parse_args(argv)


async def _main(args):
    try:
        result = await run_classify(
            SessionFactory, HttpTopicClient(args.service_url),
            batch_size=args.batch_size, limit=args.limit, dry_run=args.dry_run,
            reclassify_all=args.reclassify_all, question_ids=args.question_id,
            exclude_patterns=tuple(args.exclude_pattern),
        )
    finally:
        await engine.dispose()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(_main(parse_args(sys.argv[1:]))))


if __name__ == "__main__":
    main()
