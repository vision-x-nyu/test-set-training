"""
All protocols for bias detection models and evaluators.

This module defines the complete interface hierarchy for the TsT evaluation framework:
- BiasModel: Generic interface for any bias detection model
- FeatureBasedBiasModel: Specific interface for traditional ML models with feature engineering
- ModelEvaluator: Abstract base for evaluation strategies
"""

from typing import Protocol, Literal, Optional, List, runtime_checkable
from abc import ABC, abstractmethod
import pandas as pd

from .results import EvaluationResult, FoldResult


@runtime_checkable
class BiasModel(Protocol):
    """Base protocol for any bias detection model (RF, LLM, etc.)"""

    name: str
    format: Literal["mc", "num"]  # multiple choice or numerical
    target_col_override: Optional[str] = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset to relevant rows for this model"""
        ...

    @property
    def task(self) -> Literal["clf", "reg"]:
        """Classification or regression task"""
        ...

    @property
    def metric(self) -> Literal["acc", "mra"]:
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
    def task(self) -> Literal["clf", "reg"]:
        if self.format == "mc":
            return "clf"
        elif self.format == "num":
            return "reg"
        else:
            raise ValueError(f"Unknown format: {self.format}")

    @property
    def metric(self) -> Literal["acc", "mra"]:
        if self.format == "mc":
            return "acc"
        elif self.format == "num":
            return "mra"
        else:
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
