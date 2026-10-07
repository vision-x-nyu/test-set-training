"""
VSI-Bench (Visual Spatial Intelligence) benchmark implementation.
"""

import logging
from typing import List, Optional

import pandas as pd

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel
from ...core.qa_models import GlobalBenchmarkQAModel, MCBenchmarkQAModel, NumericalBenchmarkQAModel
from ...utils import load_hf_split
from .models import (
    FEATURE_SETS,
    GOLD_PAIR_FEATURES,
    ObjCountModel,
    ObjAbsDistModel,
    ObjSizeEstModel,
    RoomSizeEstModel,
    RelDistanceModel,
    RelDirModel,
    RoutePlanningModel,
    ObjOrderModel,
)

logger = logging.getLogger(__name__)

# HF revision whose row order the proceedings' TsT-RF number was computed on. Shuffled
# k-fold assignment depends on row order: the 2025-11-11 revision (d7cb1a3) has the same
# questions in a different order and gives slightly different TsT-RF scores.
VSI_REVISION = "bc96b17cb6be84878a6c1f2e64c24346356e0d04"
VSI_REVISION_2025_11_11 = "d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d"

# Row-order fingerprints (TsT.utils.row_order_sha1 over "id") of the test split; every load is
# checked against them.
VSI_ROW_ORDER_FINGERPRINTS = {
    VSI_REVISION: "8c48d42738671ef11a1ccb6300c81ed5d893f10f",
    VSI_REVISION_2025_11_11: "b48270fe65bfe142baa12835a230797f6e4f4ec7",
}

PAPER_PRESET_WARNING = (
    "WARNING: --feature_set paper reproduces the proceedings' VSI-Bench TsT-RF feature list "
    "(43.50 with random folds at the pinned revision). Its object_rel_distance model adds "
    f"{len(GOLD_PAIR_FEATURES)} features that are looked up with each held-out question's own gold "
    "answer, so this preset reads the held-out label. Use it only to reproduce the proceedings' "
    "number; use the default (leak-free) feature set for diagnostics."
)


@BenchmarkRegistry.register
class VSIBenchmark(Benchmark):
    """Visual Spatial Intelligence benchmark for spatial reasoning evaluation."""

    name = "vsi"
    description = "Evaluates spatial reasoning biases across numerical and multiple choice tasks"
    hf_repo = "nyu-visionx/VSI-Bench"
    default_revision = VSI_REVISION
    # The paper's TsT-LLM runs (App. C.1) loaded the 2025-11-11 revision.
    llm_revision = VSI_REVISION_2025_11_11
    id_col = "id"
    group_col_hint = "scene_name"
    feature_sets = FEATURE_SETS
    row_order_fingerprints = VSI_ROW_ORDER_FINGERPRINTS

    def load_data(self, revision: Optional[str] = None) -> pd.DataFrame:
        """Load and preprocess the VSI-Bench test split (``default_revision`` unless ``revision`` is given).

        The rows are checked against the known row-order fingerprint of the revision
        (see Benchmark.check_row_order).
        """
        revision = self.resolve_revision(revision)
        df = load_hf_split(self.hf_repo, revision)
        self.check_row_order(df, revision)
        return self.preprocess(df)

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Add gt_val / gt_idx / question_format to the raw HF test split."""
        df = df.copy()
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

    def get_feature_based_models(self, feature_set: str = "default") -> List[FeatureBasedBiasModel]:
        """Get per-question-type models for RandomForest evaluation."""
        self.check_feature_set(feature_set)
        if feature_set == "paper":
            logger.warning(PAPER_PRESET_WARNING)
        return [
            # NUM models
            ObjCountModel(),
            ObjAbsDistModel(),
            ObjSizeEstModel(),
            RoomSizeEstModel(),
            # MC models
            RelDistanceModel(feature_set=feature_set),
            RelDirModel(),
            RoutePlanningModel(),
            ObjOrderModel(),
        ]

    def get_qa_models(self) -> List[GlobalBenchmarkQAModel]:
        """TsT-LLM models: the multiple-choice and the numerical questions."""
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

    def get_ibp_strategy(self):
        """IBP strategy: keep at least 10 questions per question type."""
        from .debiasing import VSIIBPStrategy

        return VSIIBPStrategy()

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
