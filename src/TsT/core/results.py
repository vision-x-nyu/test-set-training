"""
Unified evaluation result structures for the TsT framework.

This module provides rich result objects that capture detailed information
from cross-validation evaluation, including fold-level metadata and aggregated statistics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class FoldResult:
    """Result from a single CV fold"""

    fold_id: int
    score: float
    fold_size: int
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
        return sum(f.fold_size for f in self.fold_results)

    @classmethod
    def from_fold_results(cls, repeat_id: int, fold_results: List[FoldResult]) -> "RepeatResult":
        """Create RepeatResult from fold results with calculated statistics"""
        scores = [f.score for f in fold_results]

        if scores:
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
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
    total_count: int

    # Model-specific metadata
    feature_importances: Optional[pd.DataFrame] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_repeat_results(
        cls,
        model_name: str,
        model_format: str,
        metric_name: str,
        repeat_results: List[RepeatResult],
        feature_importances: Optional[pd.DataFrame] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvaluationResult":
        """Create EvaluationResult from repeat results with calculated statistics"""
        repeat_means = [r.mean_score for r in repeat_results]

        if repeat_means:
            overall_mean = float(np.mean(repeat_means))
            overall_std = float(np.std(repeat_means))
        else:
            overall_mean = 0.0
            overall_std = 0.0

        total_count = repeat_results[0].total_instances if repeat_results else 0

        return cls(
            model_name=model_name,
            model_format=model_format,
            metric_name=metric_name,
            repeat_results=repeat_results,
            overall_mean=overall_mean,
            overall_std=overall_std,
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
            "Score": f"{self.overall_mean:.1%}",
            "± Std": f"{self.overall_std:.1%}",
            "Count": self.total_count,
            "Feature Importances": self.feature_importances,
            "Metadata": self.model_metadata,
        }

    def to_legacy_tuple(self) -> tuple[float, float, Optional[pd.DataFrame], int]:
        """Convert to legacy format for backward compatibility"""
        return (self.overall_mean, self.overall_std, self.feature_importances, self.total_count)
