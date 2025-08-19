"""
Visual Spatial Intelligence (VSI) benchmark implementation.
"""

from typing import List
import pandas as pd

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel
from ...core.qa_models import SimpleBenchmarkQAModel
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
from .data_loader import load_data as _load_data


@BenchmarkRegistry.register
class VSIBenchmark(Benchmark):
    """Visual Spatial Intelligence benchmark for spatial reasoning evaluation."""

    name = "vsi"
    description = "Evaluates spatial reasoning biases across numerical and multiple choice tasks"

    def load_data(self) -> pd.DataFrame:
        """Load VSI dataset."""
        return _load_data()

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

    def get_qa_model(self) -> SimpleBenchmarkQAModel:
        """Get single model for LLM evaluation of entire benchmark (mixed format)."""
        # Since VSI has both numerical and MC questions, we'll handle as mixed
        # But for simplicity, let's focus on MC questions for LLM evaluation
        return SimpleBenchmarkQAModel(
            benchmark_name=self.name,
            name=f"{self.name}_mc",
            format="mc",  # Focus on MC questions for LLM evaluation
            question_types=[
                "object_rel_distance",
                "object_rel_direction",
                "route_planning",
                "obj_appearance_order",
            ],  # Only MC question types
        )

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
