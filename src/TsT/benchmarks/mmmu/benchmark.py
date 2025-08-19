"""
MMMU (Massive Multi-discipline Multimodal Understanding) benchmark implementation.
"""

from typing import List
import pandas as pd

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel
from ...core.qa_models import SimpleBenchmarkQAModel
from .models import MMMUMCModel
from .data_loader import load_data as _load_data


@BenchmarkRegistry.register
class MMMUBenchmark(Benchmark):
    """MMMU benchmark for multimodal understanding evaluation."""

    name = "mmmu"
    description = "Evaluates multimodal understanding across multiple academic disciplines"

    def load_data(self) -> pd.DataFrame:
        """Load MMMU dataset."""
        return _load_data()

    def get_feature_based_models(self) -> List[FeatureBasedBiasModel]:
        """Get per-question-type models for RandomForest evaluation."""
        return [
            MMMUMCModel(),
        ]

    def get_qa_model(self) -> SimpleBenchmarkQAModel:
        """Get single model for LLM evaluation of entire benchmark."""
        return SimpleBenchmarkQAModel(
            benchmark_name=self.name,
            name=f"{self.name}_all",
            format="mc",  # All MMMU questions are multiple choice
            question_types=None,  # Evaluate all types together
        )
