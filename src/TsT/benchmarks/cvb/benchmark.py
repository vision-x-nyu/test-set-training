"""
Cambrian Vision-Centric Benchmark (CV-Bench) benchmark implementation.
"""

from typing import List
import pandas as pd
from datasets import load_dataset, Dataset

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel
from ...core.qa_models import MCBenchmarkQAModel
from .models import Count2DModel, Relation2DModel, Depth3DModel, Distance3DModel


@BenchmarkRegistry.register
class CVBBenchmark(Benchmark):
    """Cambrian Vision-Centric Benchmark for spatial reasoning evaluation."""

    name = "cvb"
    description = "Evaluates spatial reasoning biases in vision-language models"

    @staticmethod
    def load_data() -> pd.DataFrame:
        """Load and preprocess CV-Bench data."""
        testset = load_dataset("nyu-visionx/CV-Bench", split="test")
        assert isinstance(testset, Dataset), f"Expected Dataset, got {type(testset)}"

        df = testset.to_pandas(batched=False)
        assert isinstance(df, pd.DataFrame), f"Expected pd.DataFrame, got {type(df)}"

        df["question_type"] = df["task"].str.lower() + "_" + df["type"].str.lower()
        df["gt_idx"] = df["answer"].apply(lambda x: ord(x[1]) - ord("A"))
        df["gt_option"] = df.apply(lambda row: row["choices"][row["gt_idx"]], axis=1)
        df["n_options"] = df["choices"].apply(len)

        df["question_format"] = "mc"

        return df

    def get_feature_based_models(self) -> List[FeatureBasedBiasModel]:
        """Get per-question-type models for RandomForest evaluation."""
        return [
            Count2DModel(),
            Relation2DModel(),
            Depth3DModel(),
            Distance3DModel(),
        ]

    def get_qa_models(self) -> List[MCBenchmarkQAModel]:
        """Get QA models for LLM evaluation of entire benchmark.

        NOTE: CV-Bench has only multiple choice questions with 4 options.
        """
        return [MCBenchmarkQAModel(benchmark_name=self.name)]
