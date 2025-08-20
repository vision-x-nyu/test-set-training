"""
MMMU (Massive Multi-discipline Multimodal Understanding) benchmark implementation.
"""

from typing import List
import json
import os
import ast
import warnings

import pandas as pd
from datasets import load_dataset, Dataset

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel, QAFormat
from ...core.qa_models import GlobalBenchmarkQAModel, MCBenchmarkQAModel
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
    def _get_question_format(mmmu_question_type: str) -> QAFormat:
        match mmmu_question_type:
            case "multiple-choice":
                return "mc"
            case "open":
                return "oe"
            case _:
                raise ValueError(f"Unknown question type: {mmmu_question_type}")

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

        # get question format from mmmu question type
        df["question_format"] = df["question_type"].apply(MMMUBenchmark._get_question_format)

        # set "subfield_truncated" as new "question_type"
        df["question_type_original"] = df["question_type"].copy()
        df["question_type"] = df["subfield_truncated"]

        return df

    def get_feature_based_models(self) -> List[FeatureBasedBiasModel]:
        """Get per-question-type models for RandomForest evaluation."""
        return [
            MMMUMCModel(),
        ]

    def get_qa_models(self) -> List[GlobalBenchmarkQAModel]:
        """Get QA models for LLM evaluation of entire benchmark.

        NOTE: MMMU has "multiple-choice" and "open" questions.
        TODO: add open-ended support.
        """
        warnings.warn(
            "MMMU has 'multiple-choice' and 'open' questions. Only 'multiple-choice' is currently evaluated. TODO: add open-ended questions."
        )
        return [
            # Only evaluate multiple-choice questions for now
            MCBenchmarkQAModel(benchmark_name=self.name),
            # TODO: add other question types
        ]
