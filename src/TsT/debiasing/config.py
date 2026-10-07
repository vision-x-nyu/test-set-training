"""Configuration for one Iterative Bias Pruning (IBP) run on one group of questions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# What IBP ranks by:
#   "s":     the held-out TsT score s(x) itself (TsT-RF: P(gold) for MC, MRA for NUM).
#   "delta": s(x) minus the base model's zero-shot score on the same question (TsT-LLM only):
#            the per-question gain from fine-tuning on the other folds.
Score = Literal["s", "delta"]
SCORES = ("s", "delta")


@dataclass
class IBPConfig:
    """Settings for :func:`TsT.debiasing.ibp.iterative_bias_pruning`.

    Args:
        budget: Total number of questions to remove.
        batch_size: Questions removed per iteration (TsT is re-run after every batch).
        early_stop_threshold: Stop when the highest remaining bias score is at or below this value.
        score: What to rank by, "s" (raw TsT score) or "delta" (s minus zero-shot; TsT-LLM only).
        verbose: Log every iteration.
    """

    budget: int
    batch_size: int = 50
    early_stop_threshold: Optional[float] = None
    score: Score = "s"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError(f"budget must be > 0, got {self.budget}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.batch_size > self.budget:
            raise ValueError(f"batch_size ({self.batch_size}) must be <= budget ({self.budget})")
        if self.score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {self.score!r}")
