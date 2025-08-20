"""
Concrete implementations of QuestionAnswerBiasModel for LLM evaluation.
"""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
from .protocols import QAFormat, QuestionAnswerBiasModel


@dataclass
class GlobalBenchmarkQAModel(QuestionAnswerBiasModel):
    """QA model that evaluates all question types in a benchmark together."""

    name: str
    benchmark_name: str
    format: QAFormat
    question_types: Optional[List[str]] = None  # None means all types

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select rows for this benchmark, optionally filtering by question types."""
        if self.question_types:
            return df[df["question_type"].isin(self.question_types)]
        return df


@dataclass
class MCBenchmarkQAModel(GlobalBenchmarkQAModel):
    """QA model that evaluates multiple choice questions in a benchmark."""

    def __init__(self, benchmark_name: str, question_types: Optional[List[str]] = None):
        super().__init__(
            name=f"{benchmark_name}_mc", benchmark_name=benchmark_name, format="mc", question_types=question_types
        )

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["question_format"] == "mc"]


@dataclass
class NumericalBenchmarkQAModel(GlobalBenchmarkQAModel):
    """QA model that evaluates numerical questions in a benchmark."""

    def __init__(self, benchmark_name: str, question_types: Optional[List[str]] = None):
        super().__init__(
            name=f"{benchmark_name}_num", benchmark_name=benchmark_name, format="num", question_types=question_types
        )

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["question_format"] == "num"]
