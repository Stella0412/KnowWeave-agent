"""每轮指标落库 writer：把回调聚合结果写入 turn_metrics，失败只记日志不影响回答。"""

import logging

from sqlalchemy.exc import IntegrityError

from app.db.repository import MetricsRepository

logger = logging.getLogger(__name__)


class TurnMetricsWriter:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    async def write(self, payload: dict) -> None:
        try:
            async with self._session_factory() as session:
                await MetricsRepository(session).add(**payload)
                await session.commit()
        except IntegrityError:
            logger.warning("turn_metrics 已存在，跳过重复写入 token=%s",
                           payload.get("turn_token"))
        except Exception:
            logger.warning("turn_metrics 写入失败", exc_info=True)
