"""
All protocols for bias detection models and evaluators.

This module defines the complete interface hierarchy for the TsT evaluation framework:
- BiasModel: Generic interface for any bias detection model
- FeatureBasedBiasModel: Specific interface for traditional ML models with feature engineering
- ModelEvaluator: Abstract base for evaluation strategies
"""

from typing import Protocol, Optional, List, runtime_checkable, Literal, Dict, Any
from abc import ABC, abstractmethod

from dataclasses import dataclass, field
import pandas as pd
from TsT.utils import weighted_mean_std


# Type aliases for better ergonomics - define once, use everywhere
QAFormat = Literal["mc", "num", "oe"]  # multiple choice, numerical, or open-ended
Task = Literal["clf", "reg", "oe"]  # classification, regression, or open-ended
Metric = Literal["acc", "mra"]  # accuracy or mean relative accuracy

########################################################
# Results
########################################################


@dataclass
class FoldResult:
    """Result from a single CV fold"""

    fold_id: int
    score: float
    train_size: int
    test_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def test_size(self) -> int:
        return len(self.test_idx)


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

        # TODO: not doing flat scores
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


########################################################
# Models
########################################################


@runtime_checkable
class BiasModel(Protocol):
    """Base protocol for any bias detection model (RF, LLM, etc.)"""

    name: str  # e.g., "count_2d", ...
    benchmark_name: str  # e.g., "cvb", "vsi"
    format: QAFormat
    target_col_override: Optional[str] = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset to relevant rows for this model"""
        ...

    @property
    def task(self) -> Task:
        """Classification or regression task"""
        ...

    @property
    def metric(self) -> Metric:
        """Accuracy or mean relative accuracy"""
        ...

    def __str__(self) -> str:
        return f"{self.name} ({self.format})"


@runtime_checkable
class FeatureBasedBiasModel(BiasModel, Protocol):
    """
    Protocol for feature-based bias detection models (e.g., Random Forest).

    Extends BiasModel with feature engineering capabilities required for
    traditional ML approaches. This is what was previously called 'QType'.
    """

    feature_cols: List[str]

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect any statistics derived from *train* only (for leakage‑free CV)."""
        ...

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a **copy** of `df` with question‑specific feature columns added."""
        ...

    @property
    def task(self) -> Task:
        match self.format:
            case "mc":
                return "clf"
            case "num":
                return "reg"
            case "oe":
                raise ValueError("Open-ended tasks are not supported for feature-based models")
            case _:
                raise ValueError(f"Unknown format: {self.format}")

    @property
    def metric(self) -> Metric:
        match self.format:
            case "mc":
                return "acc"
            case "num":
                return "mra"
            case "oe":
                raise ValueError("Open-ended tasks are not supported for feature-based models")
            case _:
                raise ValueError(f"Unknown format: {self.format}")


@runtime_checkable
class QuestionAnswerBiasModel(BiasModel, Protocol):
    """
    Protocol for question-answer based bias detection models (e.g., LLMs).

    Unlike FeatureBasedBiasModel, these models work directly with
    question-answer pairs without feature engineering.
    """

    def prepare_instances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional preprocessing of data before evaluation"""
        return df

    @property
    def task(self) -> Task:
        match self.format:
            case "mc":
                return "clf"
            case "num":
                return "reg"
            case "oe":
                return "oe"
            case _:
                raise ValueError(f"Unknown format: {self.format}")

    @property
    def metric(self) -> Metric:
        match self.format:
            case "mc":
                return "acc"
            case "num":
                return "mra"
            case "oe":
                raise NotImplementedError("TODO")
            case _:
                raise ValueError(f"Unknown format: {self.format}")


class ModelEvaluator(ABC):
    """Abstract base for model evaluation strategies"""

    @abstractmethod
    def train_and_evaluate_fold(
        self,
        model: BiasModel,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_id: int,
        seed: int,
    ) -> FoldResult:
        """Evaluate a single fold and return score"""
        pass

    def process_results(
        self,
        model: BiasModel,
        df: pd.DataFrame,
        target_col: str,
        evaluation_result: EvaluationResult,
    ) -> EvaluationResult:
        """Default results processor that simply returns unmodified evaluation result."""
        return evaluation_result
