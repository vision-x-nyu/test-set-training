"""
Base protocols for bias detection models and evaluators.

This module defines the fundamental interfaces that all bias detection models
and evaluation strategies must implement to work with the unified evaluation framework.
"""

from typing import Protocol, Literal, Optional, runtime_checkable
from abc import ABC, abstractmethod
import pandas as pd


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


class ModelEvaluator(ABC):
    """Abstract base for model evaluation strategies"""

    @abstractmethod
    def evaluate_fold(
        self, model: BiasModel, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, fold_num: int, seed: int
    ) -> float:
        """Evaluate a single fold and return score"""
        pass
