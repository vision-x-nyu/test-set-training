"""
Selection strategies for Iterative Bias Pruning (IBP).

TopKDiversityStrategy removes the highest-scoring questions while keeping a minimum number
of questions in every group (e.g. question type). Benchmark-specific strategies
(``TsT.benchmarks.<name>.debiasing``) subclass it with their grouping column and floor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import pandas as pd

TieBreak = Literal["id", "legacy"]
TIE_BREAKS = ("id", "legacy")
# Precision at which tie_break="id" compares scores (far below any real score difference,
# far above floating-point summation noise).
SCORE_DECIMALS = 12


class MissingScoreError(KeyError):
    """A question that IBP should rank has no score."""

    def __str__(self) -> str:  # KeyError would print the repr of the message
        return str(self.args[0]) if self.args else ""


class TopKDiversityStrategy:
    """Greedy top-k removal that preserves a minimum group size.

    The bias score of a question is the score IBP passes in (``sample_predictions``). Candidates
    are visited in descending score order; a candidate is removed only if its group keeps more
    than ``min_samples_per_group`` questions afterwards.

    Ties (common: hundreds of numerical questions score exactly 1.0, and Random Forest
    probabilities take few distinct values) are broken deterministically:

    - ``tie_break="id"`` (default): stable sort on (-score, ``id_col`` value), falling back to the
      DataFrame index label when ``id_col`` is unset or absent. Scores are compared after rounding
      to ``SCORE_DECIMALS`` decimals, so floating-point noise in the last bits (e.g. from the order
      in which a parallel Random Forest sums its trees) cannot reorder ties. The selection does not
      depend on row order.
    - ``tie_break="legacy"``: pandas' default (unstable) sort on the score alone, as in the
      proceedings' runs. The order among tied scores then depends on row order and on the
      numpy sort implementation; use it only to re-derive those removal sets.

    Args:
        group_column: Column whose groups must keep ``min_samples_per_group`` questions.
        min_samples_per_group: Minimum group size after removal.
        id_col: Question-id column used to break ties (IBP sets it to the benchmark's id column
            when left as None).
        tie_break: "id" or "legacy" (see above).
    """

    def __init__(
        self,
        group_column: str,
        min_samples_per_group: int = 5,
        id_col: Optional[str] = None,
        tie_break: TieBreak = "id",
    ) -> None:
        if tie_break not in TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {TIE_BREAKS}, got {tie_break!r}")
        if min_samples_per_group < 0:
            raise ValueError(f"min_samples_per_group must be >= 0, got {min_samples_per_group}")
        self.group_column = group_column
        self.min_samples_per_group = min_samples_per_group
        self.id_col = id_col
        self.tie_break = tie_break

    def compute_bias_scores(self, df: pd.DataFrame, sample_predictions: Dict[Any, float]) -> pd.Series:
        """Bias score of every row of ``df``: its entry in ``sample_predictions``.

        Raises:
            MissingScoreError: if a row of ``df`` has no score.
        """
        missing = [idx for idx in df.index if idx not in sample_predictions]
        if missing:
            raise MissingScoreError(
                f"{len(missing)} of {len(df)} questions to rank have no score (first index labels: "
                f"{missing[:5]}). Every question IBP ranks must be scored by a model."
            )
        return pd.Series([float(sample_predictions[idx]) for idx in df.index], index=df.index, dtype=float)

    def ranking(self, df: pd.DataFrame, bias_scores: pd.Series) -> pd.Index:
        """Index labels of ``bias_scores`` from most to least biased, ties broken by ``tie_break``."""
        if self.tie_break == "legacy":
            return bias_scores.sort_values(ascending=False).index
        if self.id_col is not None and self.id_col in df.columns:
            key = df.loc[bias_scores.index, self.id_col]
        else:
            key = bias_scores.index.to_series(index=bias_scores.index)
        order = pd.DataFrame({"score": bias_scores.round(SCORE_DECIMALS), "key": key}, index=bias_scores.index)
        return order.sort_values(["score", "key"], ascending=[False, True], kind="mergesort").index

    def select_removal_candidates(self, df: pd.DataFrame, bias_scores: pd.Series, batch_size: int) -> List[Any]:
        """Up to ``batch_size`` index labels to remove, preserving the group floor.

        ``bias_scores`` may cover only part of ``df`` (questions no model scores are never
        removed); group sizes are counted over all of ``df``.
        """
        if len(df) == 0 or len(bias_scores) == 0:
            return []
        group_counts = df[self.group_column].value_counts().to_dict()
        candidates: List[Any] = []
        for idx in self.ranking(df, bias_scores):
            if len(candidates) >= batch_size:
                break
            group = df.loc[idx, self.group_column]
            if group_counts.get(group, 0) > self.min_samples_per_group:
                candidates.append(idx)
                group_counts[group] -= 1
        return candidates
