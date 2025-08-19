"""
All protocols for bias detection models and evaluators.

This module defines the complete interface hierarchy for the TsT evaluation framework:
- BiasModel: Generic interface for any bias detection model
- FeatureBasedBiasModel: Specific interface for traditional ML models with feature engineering
- ModelEvaluator: Abstract base for evaluation strategies
"""

from typing import Protocol, Optional, List, runtime_checkable, Literal
from abc import ABC, abstractmethod
import pandas as pd

from .results import EvaluationResult, FoldResult


# Type aliases for better ergonomics - define once, use everywhere
QAFormat = Literal["mc", "num", "oe"]  # multiple choice, numerical, or open-ended
Task = Literal["clf", "reg", "oe"]  # classification, regression, or open-ended
Metric = Literal["acc", "mra"]  # accuracy or mean relative accuracy


@runtime_checkable
class BiasModel(Protocol):
    """Base protocol for any bias detection model (RF, LLM, etc.)"""

    name: str
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

    benchmark_name: str  # e.g., "cvb", "vsi"

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
