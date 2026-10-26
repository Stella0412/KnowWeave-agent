from pydantic import BaseModel


class LabelCount(BaseModel):
    label: str
    count: int
    share: float


class ComboCount(BaseModel):
    labels: list[str]
    count: int


class DistributionResponse(BaseModel):
    window_days: int
    weighted: bool
    total_roots: int
    total_weighted: int
    unclassified_roots: int
    last_classified_at: str | None
    labels: list[LabelCount]
    combos: list[ComboCount]
