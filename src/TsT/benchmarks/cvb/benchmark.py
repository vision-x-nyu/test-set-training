"""
Compositional Visual Biases (CVB) benchmark implementation.
"""

from typing import List
import pandas as pd

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel
from ...core.qa_models import SimpleBenchmarkQAModel
from .models import Count2DModel, Relation2DModel, Depth3DModel, Distance3DModel
from .data_loader import load_data as _load_data


@BenchmarkRegistry.register
class CVBBenchmark(Benchmark):
    """Compositional Visual Biases benchmark for spatial reasoning evaluation."""

    name = "cvb"
    description = "Evaluates spatial reasoning biases in vision-language models"

    def load_data(self) -> pd.DataFrame:
        """Load CVB dataset."""
        return _load_data()

    def get_feature_based_models(self) -> List[FeatureBasedBiasModel]:
        """Get per-question-type models for RandomForest evaluation."""
        return [
            Count2DModel(),
            Relation2DModel(),
            Depth3DModel(),
            Distance3DModel(),
        ]

    def get_qa_model(self) -> SimpleBenchmarkQAModel:
        """Get single model for LLM evaluation of entire benchmark."""
        return SimpleBenchmarkQAModel(
            benchmark_name=self.name,
            name=f"{self.name}_all",
            format="mc",  # All CVB questions are multiple choice
            question_types=None,  # Evaluate all types together
        )
