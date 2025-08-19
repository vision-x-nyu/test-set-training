"""
MMMU (Massive Multi-discipline Multimodal Understanding) benchmark implementation.
"""

from typing import List
import json
import os
import ast
import pandas as pd
from datasets import load_dataset, Dataset

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel
from ...core.qa_models import SimpleBenchmarkQAModel
from .models import MMMUMCModel


# get path to this file
this_dir = os.path.dirname(os.path.abspath(__file__))
subfield_cluster_map = json.load(open(os.path.join(this_dir, "subfield_cluster_map.json")))


@BenchmarkRegistry.register
class MMMUBenchmark(Benchmark):
    """MMMU benchmark for multimodal understanding evaluation."""

    name = "mmmu"
    description = "Evaluates multimodal understanding across multiple academic disciplines"

    @staticmethod
    def load_data() -> pd.DataFrame:
        """Load and preprocess MMMU validation data."""
        valset = load_dataset("lmms-lab/MMMU", split="validation")
        assert isinstance(valset, Dataset), f"Expected Dataset, got {type(valset)}"

        df = valset.to_pandas(batched=False)
        assert isinstance(df, pd.DataFrame), f"Expected pd.DataFrame, got {type(df)}"

        # rename answer to ground_truth
        df = df.rename(columns={"answer": "ground_truth"})

        # Parse options from string to list
        df["options"] = df["options"].apply(ast.literal_eval)
        df["num_options"] = df["options"].apply(len)
        df["img_type"] = df["img_type"].apply(ast.literal_eval)

        # group all subfields with fewer than 5 examples into "Other"
        df["subfield_truncated"] = (
            df.groupby("subfield")["subfield"].transform(lambda x: "Other" if len(x) < 5 else x).fillna("Other")
        )

        df["subfield_cluster"] = df["subfield"].map(subfield_cluster_map)

        # Extract the ground truth answer index (A=0, B=1, ...)
        df["gt_idx"] = df["ground_truth"].apply(lambda x: "ABCDEFGHI".index(x) if x in "ABCDEFGHI" else -1)

        # Extract the ground truth answer text
        df["gt_val"] = df.apply(
            lambda row: row["options"][row["gt_idx"]] if 0 <= row["gt_idx"] < len(row["options"]) else None, axis=1
        )

        return df

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
