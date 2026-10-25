from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_current: ContextVar["TurnObserver | None"] = ContextVar("ch09_turn_observer", default=None)


@dataclass
class TurnObserver:
    handler: object
    turn_token: str

    def record_stage(self, kind, *, model, duration_ms, texts=None, items=0):
        try:
            self.handler.record_stage(
                self.turn_token, kind, model=model,
                duration_ms=duration_ms, texts=texts, items=items,
            )
        except Exception:
            pass


def get_observer():
    return _current.get()


@contextmanager
def observer_scope(observer):
    token = _current.set(observer)
    try:
        yield
    finally:
        _current.reset(token)
