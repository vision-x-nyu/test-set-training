"""
Unified evaluation result structures for the TsT framework.

This module provides rich result objects that capture detailed information
from cross-validation evaluation, including fold-level metadata and aggregated statistics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd

from TsT.utils import weighted_mean_std


@dataclass
class FoldResult:
    """Result from a single CV fold"""

    fold_id: int
    score: float
    train_size: int
    test_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepeatResult:
    """Result from a single repeat (collection of folds)"""

    repeat_id: int
    fold_results: List[FoldResult]
    mean_score: float
    std_score: float

    @property
    def total_instances(self) -> int:
        return sum(f.test_size for f in self.fold_results)

    @classmethod
    def from_fold_results(cls, repeat_id: int, fold_results: List[FoldResult]) -> "RepeatResult":
        """Create RepeatResult from fold results with calculated statistics"""

        scores = [f.score for f in fold_results]
        counts = [f.test_size for f in fold_results]

        if scores and counts:
            mean_score, std_score = weighted_mean_std(scores, counts)
        else:
            mean_score = 0.0
            std_score = 0.0

        return cls(
            repeat_id=repeat_id,
            fold_results=fold_results,
            mean_score=mean_score,
            std_score=std_score,
        )


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model"""

    model_name: str
    model_format: str
    metric_name: str
    repeat_results: List[RepeatResult]

    # Aggregated statistics
    overall_mean: float
    overall_std: float
    count: int
    repeats: int
    total_count: int

    # Model-specific metadata
    zero_shot_baseline: float = 0.0
    feature_importances: Optional[pd.DataFrame] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_repeat_results(
        cls,
        model_name: str,
        model_format: str,
        metric_name: str,
        repeat_results: List[RepeatResult],
        zero_shot_baseline: float = 0.0,
        feature_importances: Optional[pd.DataFrame] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvaluationResult":
        """Create EvaluationResult from repeat results with calculated statistics"""
        # repeat_means = [r.mean_score for r in repeat_results]

        assert len(repeat_results) > 0, "Must have at least one repeat"
        assert all(len(repeat.fold_results) > 0 for repeat in repeat_results), "Each repeat must have at least one fold"

        flat_scores = [fold.score for repeat in repeat_results for fold in repeat.fold_results]
        flat_counts = [fold.test_size for repeat in repeat_results for fold in repeat.fold_results]

        if flat_scores and flat_counts:
            overall_mean, overall_std = weighted_mean_std(flat_scores, flat_counts)
        else:
            overall_mean = 0.0
            overall_std = 0.0

        total_count = sum(flat_counts)
        repeats = len(repeat_results)
        count = total_count // repeats
        assert count == repeat_results[0].total_instances, (
            f"Overall count {count} does not match repeat {repeat_results[0].repeat_id} total instances {repeat_results[0].total_instances}"
        )

        return cls(
            model_name=model_name,
            model_format=model_format,
            metric_name=metric_name,
            repeat_results=repeat_results,
            zero_shot_baseline=zero_shot_baseline,
            overall_mean=overall_mean,
            overall_std=overall_std,
            count=count,
            repeats=repeats,
            total_count=total_count,
            feature_importances=feature_importances,
            model_metadata=model_metadata or {},
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for summary table"""
        return {
            "Model": self.model_name,
            "Format": self.model_format.upper(),
            "Metric": self.metric_name.upper(),
            "Zero-shot Baseline": self.zero_shot_baseline,
            "Score": self.overall_mean,
            "± Std": self.overall_std,
            "Count": self.count,
            "Repeats": self.repeats,
            "Total Count": self.total_count,
            "Feature Importances": self.feature_importances,
            "Metadata": self.model_metadata,
        }
