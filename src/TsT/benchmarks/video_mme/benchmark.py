"""
Video-MME (Video Multimodal Evaluation) benchmark implementation.
"""

from typing import List
import pandas as pd

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel
from ...core.qa_models import SimpleBenchmarkQAModel
from .models import VideoMMEModel
from .data_loader import load_data as _load_data


@BenchmarkRegistry.register
class VideoMMEBenchmark(Benchmark):
    """Video-MME benchmark for video understanding evaluation."""

    name = "video_mme"
    description = "Evaluates video understanding across multiple domains and task types"

    def load_data(self) -> pd.DataFrame:
        """Load Video-MME dataset."""
        return _load_data()

    def get_feature_based_models(self) -> List[FeatureBasedBiasModel]:
        """Get per-question-type models for RandomForest evaluation."""
        return [
            VideoMMEModel(),  # Global model for all video questions
        ]

    def get_qa_model(self) -> SimpleBenchmarkQAModel:
        """Get single model for LLM evaluation of entire benchmark."""
        return SimpleBenchmarkQAModel(
            benchmark_name=self.name,
            name=f"{self.name}_all",
            format="mc",  # All Video-MME questions are multiple choice
            question_types=None,  # Evaluate all types together
        )
