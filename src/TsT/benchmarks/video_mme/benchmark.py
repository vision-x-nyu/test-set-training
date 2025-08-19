"""
Video-MME (Video Multimodal Evaluation) benchmark implementation.
"""

from typing import List
import pandas as pd
from datasets import load_dataset, Dataset

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel
from ...core.qa_models import GlobalBenchmarkQAModel
from .models import VideoMMEModel


@BenchmarkRegistry.register
class VideoMMEBenchmark(Benchmark):
    """Video-MME benchmark for video understanding evaluation."""

    name = "video_mme"
    description = "Evaluates video understanding across multiple domains and task types"

    @staticmethod
    def load_data() -> pd.DataFrame:
        """Load and preprocess Video-MME data."""
        testset = load_dataset("lmms-lab/Video-MME", split="test")
        assert isinstance(testset, Dataset), f"Expected Dataset, got {type(testset)}"

        df = testset.to_pandas(batched=False)
        assert isinstance(df, pd.DataFrame), f"Expected pd.DataFrame, got {type(df)}"

        # Video-MME has multiple choice questions with 4 options
        # Extract the ground truth answer index (A=0, B=1, C=2, D=3)
        df["gt_idx"] = df["answer"].apply(lambda x: "ABCD".index(x))

        # Extract the ground truth answer text
        df["gt_val"] = df["answer"]

        df["question_type"] = df["task_type"]

        return df

    def get_feature_based_models(self) -> List[FeatureBasedBiasModel]:
        """Get per-question-type models for RandomForest evaluation."""
        return [
            VideoMMEModel(),  # Global model for all video questions
        ]

    def get_qa_model(self) -> GlobalBenchmarkQAModel:
        """Get single model for LLM evaluation of entire benchmark."""
        return GlobalBenchmarkQAModel(
            benchmark_name=self.name,
            name=f"{self.name}_all",
            format="mc",  # All Video-MME questions are multiple choice
            question_types=None,  # Evaluate all types together
        )
