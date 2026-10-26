from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from app.db.repository import TopicRepository


class TopicAdminService:
    def __init__(self, session_factory, taxonomy_path="config/ch10_taxonomy.json"):
        self._session_factory = session_factory
        raw = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
        self._names = tuple(item["name"] for item in raw["labels"])
        if len(self._names) != 17 or self._names[-1] != "其他":
            raise ValueError("类目表必须是 17 项且『其他』最后")

    async def distribution(self, days: int = 30, weighted: bool = False) -> dict:
        since = datetime.now() - timedelta(days=days)
        async with self._session_factory() as session:
            repo = TopicRepository(session)
            dist = await repo.distribution(names=self._names, since=since, weighted=weighted)
            coverage = await repo.coverage()
            combos = await repo.combos(since=since)
        return {
            "window_days": days,
            "weighted": weighted,
            "total_roots": dist["total_roots"],
            "total_weighted": dist["total_weighted"],
            "unclassified_roots": coverage["unclassified_roots"],
            "last_classified_at": coverage["last_classified_at"],
            "labels": dist["labels"],
            "combos": combos,
        }

    async def questions(self, label: str, limit: int = 20) -> list[dict]:
        if label not in self._names:
            raise ValueError(f"未知标签: {label}")
        async with self._session_factory() as session:
            return await TopicRepository(session).questions_by_label(label, limit)
