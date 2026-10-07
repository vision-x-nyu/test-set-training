"""Result records for Iterative Bias Pruning (IBP)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import IBPConfig


@dataclass
class IBPIterationResult:
    """One IBP iteration. Bias statistics are over the questions ranked at the start of the iteration."""

    iteration: int
    removed_indices: List[Any]  # DataFrame index labels (positions in the loaded revision)
    removed_ids: List[Any]  # the benchmark's question ids of the same questions
    max_bias_score: float
    mean_bias_score: float
    n_ranked: int
    dataset_size_before: int
    dataset_size_after: int


@dataclass
class IBPResult:
    """One IBP run on one group of questions."""

    removed_indices: List[Any]
    removed_ids: List[Any]
    iterations: List[IBPIterationResult]
    config: IBPConfig
    original_size: int
    final_size: int
    stop_reason: str = "budget"  # "budget", "early_stop" or "no_candidates"

    @property
    def removal_rate(self) -> float:
        return len(self.removed_indices) / self.original_size if self.original_size > 0 else 0.0

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupResult:
    """IBP result for one allocation group (e.g. one answer format)."""

    name: str
    size: int
    n_ranked: int  # questions some model scores; the others are never removed
    budget: int
    models: List[str]
    result: IBPResult

    def summary(self) -> Dict[str, Any]:
        its = self.result.iterations
        return {
            "group": self.name,
            "n": self.size,
            "n_ranked": self.n_ranked,
            "models": self.models,
            "budget": self.budget,
            "batch_size": self.result.config.batch_size,
            "removed": len(self.result.removed_ids),
            "n_iterations": len(its),
            "stop_reason": self.result.stop_reason,
            # bias statistics at the start of the first and of the last iteration
            "max_bias_first": its[0].max_bias_score if its else None,
            "mean_bias_first": its[0].mean_bias_score if its else None,
            "max_bias_last_iteration": its[-1].max_bias_score if its else None,
            "mean_bias_last_iteration": its[-1].mean_bias_score if its else None,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "size_before": it.dataset_size_before,
                    "n_ranked": it.n_ranked,
                    "max_bias": it.max_bias_score,
                    "mean_bias": it.mean_bias_score,
                    "removed_ids": it.removed_ids,
                }
                for it in its
            ],
        }


@dataclass
class IBPRun:
    """A complete IBP run on a benchmark: settings, per-group results, removed and kept ids."""

    settings: Dict[str, Any]
    groups: List[GroupResult]
    removed_ids: List[Any]  # in removal order (group by group, iteration by iteration)
    kept_ids: List[Any]  # in dataset row order
    n_rows: int
    environment: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_removed(self) -> int:
        return len(self.removed_ids)

    def summary(self) -> Dict[str, Any]:
        return _json_safe(
            {
                **self.settings,
                "n_rows": self.n_rows,
                "n_removed": self.n_removed,
                "n_kept": len(self.kept_ids),
                "removal_rate": self.n_removed / self.n_rows if self.n_rows else None,
                "groups": [g.summary() for g in self.groups],
                "environment": self.environment,
            }
        )

    def save(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Write removed_ids.txt, kept_ids.txt (one id per line) and summary.json."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "removed_ids": out / "removed_ids.txt",
            "kept_ids": out / "kept_ids.txt",
            "summary": out / "summary.json",
        }
        paths["removed_ids"].write_text("".join(f"{i}\n" for i in self.removed_ids), encoding="utf-8")
        paths["kept_ids"].write_text("".join(f"{i}\n" for i in self.kept_ids), encoding="utf-8")
        with open(paths["summary"], "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
        return paths


def _json_safe(obj: Optional[Any]) -> Any:
    """NaN/inf -> None, numpy scalars -> Python scalars, tuples -> lists (strict JSON)."""
    if hasattr(obj, "item") and not isinstance(obj, (list, dict, str)):
        try:
            obj = obj.item()
        except (TypeError, ValueError):
            pass
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj
