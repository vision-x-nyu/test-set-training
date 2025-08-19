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
