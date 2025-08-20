"""
Visual Spatial Intelligence (VSI) benchmark implementation.
"""

from typing import List
import pandas as pd
from datasets import load_dataset, Dataset

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel
from ...core.qa_models import GlobalBenchmarkQAModel, MCBenchmarkQAModel, NumericalBenchmarkQAModel
from .models import (
    ObjCountModel,
    ObjAbsDistModel,
    ObjSizeEstModel,
    RoomSizeEstModel,
    RelDistanceModel,
    RelDirModel,
    RoutePlanningModel,
    ObjOrderModel,
)


@BenchmarkRegistry.register
class VSIBenchmark(Benchmark):
    """Visual Spatial Intelligence benchmark for spatial reasoning evaluation."""

    name = "vsi"
    description = "Evaluates spatial reasoning biases across numerical and multiple choice tasks"

    @staticmethod
    def load_data() -> pd.DataFrame:
        """Load and preprocess VSI-Bench data."""
        testset = load_dataset("nyu-visionx/VSI-Bench", split="test")
        assert isinstance(testset, Dataset), f"Expected Dataset, got {type(testset)}"

        df = testset.to_pandas(batched=False)
        assert isinstance(df, pd.DataFrame), f"Expected pd.DataFrame, got {type(df)}"

        # For numerical questions (no options)
        df["gt_val"] = df["ground_truth"]
        df["gt_idx"] = -1

        # For multiple choice questions (with options)
        mc_mask = df["options"].notna()
        df.loc[mc_mask, "gt_idx"] = df.loc[mc_mask, "ground_truth"].apply(lambda x: "ABCD".index(x))
        df.loc[mc_mask, "gt_val"] = df[mc_mask].apply(
            lambda row: row["options"][int(row["gt_idx"])].split(". ")[-1], axis=1
        )

        df["question_format"] = "num"
        df.loc[mc_mask, "question_format"] = "mc"

        return df

    def get_feature_based_models(self) -> List[FeatureBasedBiasModel]:
        """Get per-question-type models for RandomForest evaluation."""
        return [
            # NUM models
            ObjCountModel(),
            ObjAbsDistModel(),
            ObjSizeEstModel(),
            RoomSizeEstModel(),
            # MC models
            RelDistanceModel(),
            RelDirModel(),
            RoutePlanningModel(),
            ObjOrderModel(),
        ]

    def get_qa_models(self) -> List[GlobalBenchmarkQAModel]:
        return [
            MCBenchmarkQAModel(
                benchmark_name=self.name,
                question_types=[
                    "object_rel_distance",
                    "object_rel_direction",
                    "route_planning",
                    "obj_appearance_order",
                ],
            ),
            NumericalBenchmarkQAModel(
                benchmark_name=self.name,
                question_types=[
                    "object_counting",
                    "object_abs_distance",
                    "object_size_estimation",
                    "room_size_estimation",
                ],
            ),
        ]

    def get_metadata(self) -> dict:
        """Override to provide VSI-specific metadata."""
        base_metadata = super().get_metadata()

        # Add VSI-specific information
        base_metadata.update(
            {
                "num_numerical_models": 4,
                "num_mc_models": 4,
                "numerical_question_types": [
                    "object_counting",
                    "object_abs_distance",
                    "object_size_estimation",
                    "room_size_estimation",
                ],
                "mc_question_types": [
                    "object_rel_distance",
                    "object_rel_direction",
                    "route_planning",
                    "obj_appearance_order",
                ],
            }
        )

        return base_metadata
