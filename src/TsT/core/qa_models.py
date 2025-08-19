"""
Concrete implementations of QuestionAnswerBiasModel for LLM evaluation.
"""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
from .protocols import QAFormat, Task, Metric


@dataclass
class SimpleBenchmarkQAModel:
    """Simple QA model that evaluates all question types in a benchmark together."""

    benchmark_name: str
    name: str
    format: QAFormat
    question_types: Optional[List[str]] = None  # None means all types
    target_col_override: Optional[str] = None

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select rows for this benchmark, optionally filtering by question types."""
        if self.question_types:
            return df[df["question_type"].isin(self.question_types)]
        return df

    def prepare_instances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Default implementation - no preprocessing needed."""
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
