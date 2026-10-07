"""
Question-answer models for TsT-LLM: one model per answer format of a benchmark.
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .protocols import QAFormat, QuestionAnswerBiasModel


@dataclass
class GlobalBenchmarkQAModel(QuestionAnswerBiasModel):
    """Scores all of a benchmark's questions of one format together."""

    name: str
    benchmark_name: str
    format: QAFormat
    question_types: Optional[List[str]] = None  # informational
    # Column the LLM is trained to produce; a gt_idx target is rendered as the option letter.
    default_target_col: str = "ground_truth"

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.question_types:
            return df[df["question_type"].isin(self.question_types)]
        return df


class MCBenchmarkQAModel(GlobalBenchmarkQAModel):
    """The benchmark's multiple-choice questions."""

    def __init__(
        self, benchmark_name: str, question_types: Optional[List[str]] = None, default_target_col: str = "ground_truth"
    ):
        super().__init__(
            name=f"{benchmark_name}_mc",
            benchmark_name=benchmark_name,
            format="mc",
            question_types=question_types,
            default_target_col=default_target_col,
        )

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["question_format"] == "mc"]


class NumericalBenchmarkQAModel(GlobalBenchmarkQAModel):
    """The benchmark's numerical questions."""

    def __init__(self, benchmark_name: str, question_types: Optional[List[str]] = None):
        super().__init__(
            name=f"{benchmark_name}_num",
            benchmark_name=benchmark_name,
            format="num",
            question_types=question_types,
            default_target_col="ground_truth",
        )

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["question_format"] == "num"]
