"""
How an IBP budget is split into independently pruned groups of questions.

- ``global``: one group (all questions), one budget.
- ``per_format``: one group per answer format (``question_format``: mc, num, ...); a total budget is
  split in proportion to group size, rounding by largest remainder (or explicit per-format budgets).
- ``per_type``: one group per question type, each with its own budget: a uniform fraction of the
  group (``frac``) or explicit per-type counts (e.g. the per-type composition of an existing
  removal list, see :func:`budgets_from_ids`). Each budget is capped so that the strategy's
  group floor can still be met.

Each group is pruned on its own: TsT is run on that group's questions only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Mapping, Optional

import pandas as pd

Alloc = Literal["global", "per_format", "per_type"]
ALLOCS = ("global", "per_format", "per_type")
FORMAT_COL = "question_format"
TYPE_COL = "question_type"


@dataclass
class Group:
    """One independently pruned group: its rows (index labels) and removal budget."""

    name: str
    index: pd.Index
    budget: int

    @property
    def size(self) -> int:
        return len(self.index)


def proportional_budgets(counts: Mapping[str, int], budget: int) -> Dict[str, int]:
    """Split ``budget`` across groups in proportion to ``counts`` (largest-remainder rounding).

    Ties in the remainder keep the order of ``counts``.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("cannot split a budget over empty groups")
    raw = {g: budget * n / total for g, n in counts.items()}
    out = {g: int(math.floor(b)) for g, b in raw.items()}
    leftover = budget - sum(out.values())
    for g in sorted(counts, key=lambda g: raw[g] - out[g], reverse=True)[:leftover]:
        out[g] += 1
    return out


def uniform_budgets(counts: Mapping[str, int], frac: float) -> Dict[str, int]:
    """``round(frac * n)`` questions from every group."""
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    return {g: int(round(frac * n)) for g, n in counts.items()}


def budgets_from_ids(df: pd.DataFrame, ids: Iterable, id_col: str, group_col: str = TYPE_COL) -> Dict[str, int]:
    """Per-group counts of ``ids`` (e.g. a released removal list) in ``df``.

    ``ids`` are compared as strings with ``df[id_col]``. Raises ValueError if an id is not in ``df``.
    """
    wanted = {str(i).strip() for i in ids if str(i).strip()}
    present = df[id_col].astype(str)
    unknown = wanted - set(present)
    if unknown:
        raise ValueError(f"{len(unknown)} ids are not in the benchmark (e.g. {sorted(unknown)[:5]})")
    counts = df.loc[present.isin(wanted), group_col].value_counts()
    return {str(g): int(n) for g, n in counts.items()}


def _check_budgets(budgets: Mapping[str, int], counts: Mapping[str, int]) -> None:
    unknown = set(budgets) - set(counts)
    if unknown:
        raise ValueError(f"budgets name unknown groups: {sorted(unknown)} (groups: {sorted(counts)})")
    for g, b in budgets.items():
        if b < 0 or b > counts[g]:
            raise ValueError(f"budget for {g!r} must be between 0 and {counts[g]}, got {b}")


def plan_groups(
    df: pd.DataFrame,
    alloc: Alloc,
    *,
    budget: Optional[int] = None,
    frac: Optional[float] = None,
    budgets: Optional[Mapping[str, int]] = None,
    min_per_group: int = 0,
) -> List[Group]:
    """Groups to prune and their budgets.

    Args:
        df: The benchmark questions.
        alloc: "global" (needs ``budget``), "per_format" (``budget`` or ``budgets``) or "per_type"
            (``frac`` or ``budgets``).
        budget: Total budget (global, per_format).
        frac: Removal fraction per question type (per_type).
        budgets: Explicit budget per group (per-format: format name; per_type: question type);
            groups not listed get 0.
        min_per_group: The strategy's group floor; per_type budgets are capped at ``n - min_per_group``.

    Returns:
        Groups with a positive budget. Order: per_format by format name, per_type by decreasing
        size (the order only affects the order of the removal list, not which questions are removed).
    """
    if alloc not in ALLOCS:
        raise ValueError(f"alloc must be one of {ALLOCS}, got {alloc!r}")
    if alloc == "global":
        if budget is None or frac is not None or budgets is not None:
            raise ValueError("alloc='global' takes a total budget (and no frac or per-group budgets)")
    if alloc == "per_format" and ((budget is None) == (budgets is None) or frac is not None):
        raise ValueError("alloc='per_format' takes exactly one of a total budget or per-format budgets (and no frac)")
    if budget is not None:
        if budget <= 0:
            raise ValueError(f"budget must be > 0, got {budget}")
        if budget > len(df):
            raise ValueError(f"budget ({budget}) exceeds the number of questions ({len(df)})")
    if alloc == "global":
        return [Group("all", df.index, int(budget))]

    if alloc == "per_format":
        if FORMAT_COL not in df.columns:
            raise ValueError(f"alloc='per_format' needs a {FORMAT_COL!r} column")
        counts = df[FORMAT_COL].value_counts().to_dict()
        names = sorted(counts)
        if budgets is not None:
            _check_budgets(budgets, counts)
            split = {f: int(budgets.get(f, 0)) for f in names}
        else:
            split = proportional_budgets({f: counts[f] for f in names}, int(budget))
        return [Group(f, df.index[df[FORMAT_COL] == f], split[f]) for f in names if split[f] > 0]

    # per_type
    if TYPE_COL not in df.columns:
        raise ValueError(f"alloc='per_type' needs a {TYPE_COL!r} column")
    if (frac is None) == (budgets is None) or budget is not None:
        raise ValueError("alloc='per_type' takes exactly one of frac or per-type budgets (and no total budget)")
    counts = df[TYPE_COL].value_counts().to_dict()
    names = sorted(counts, key=lambda t: -counts[t])
    if frac is not None:
        planned = uniform_budgets(counts, frac)
    else:
        _check_budgets(budgets, counts)
        planned = {t: int(budgets.get(t, 0)) for t in names}
    groups = []
    for t in names:
        b = min(planned.get(t, 0), max(0, counts[t] - min_per_group))
        if b > 0:
            groups.append(Group(str(t), df.index[df[TYPE_COL] == t], b))
    return groups
