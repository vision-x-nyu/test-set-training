"""
Cambrian Vision-Centric Benchmark (CV-Bench) benchmark implementation.
"""

from typing import List, Optional

import pandas as pd

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import FeatureBasedBiasModel
from ...core.qa_models import MCBenchmarkQAModel
from ...utils import load_hf_split
from .models import Count2DModel, Relation2DModel, Depth3DModel, Distance3DModel

# HF revision used for the reported CV-Bench TsT-RF numbers.
CVB_REVISION = "bc284db50d036958861cb60cdd7b77612052ce0d"

# Row-order fingerprint (TsT.utils.row_order_sha1 over "idx") of the test split at CVB_REVISION;
# every load at that revision is checked against it.
CVB_ROW_ORDER_FINGERPRINTS = {CVB_REVISION: "02535f1c609db8bba7c6bd64d9a78b9aa3d0fa3a"}

PAPER_PRESET_UNAVAILABLE = (
    "--feature_set paper is defined only for VSI-Bench (--benchmark vsi). The CV-Bench TsT-RF score "
    "printed in the COLM 2026 proceedings (75.5) is withdrawn: it could not be reproduced, and its "
    "features read held-out labels. Use the default (leak-free) feature set, which gives 56.14 with "
    "random folds (55.65 with --group_col image_id) at the pinned revision. See docs/reproducing.md "
    "(README: 'Reproducing the paper')."
)


@BenchmarkRegistry.register
class CVBBenchmark(Benchmark):
    """Cambrian Vision-Centric Benchmark for spatial reasoning evaluation."""

    name = "cvb"
    description = "Evaluates spatial reasoning biases in vision-language models"
    hf_repo = "nyu-visionx/CV-Bench"
    default_revision = CVB_REVISION
    id_col = "idx"
    group_col_hint = "image_id"
    feature_sets = ("default",)
    row_order_fingerprints = CVB_ROW_ORDER_FINGERPRINTS

    def load_data(self, revision: Optional[str] = None) -> pd.DataFrame:
        """Load and preprocess the CV-Bench test split (``default_revision`` unless ``revision`` is given).

        NOTE: the CV-Bench test split stores its images inline, so the first load
        downloads the image parquet files (about 405 MB); only the text columns are read.
        The rows are checked against the known row-order fingerprint of the revision
        (see Benchmark.check_row_order).
        """
        revision = self.resolve_revision(revision)
        df = load_hf_split(self.hf_repo, revision, drop_columns=("image",))
        self.check_row_order(df, revision)
        return self.preprocess(df)

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Add question_type / gt_idx / gt_option / n_options / image_id to the raw HF test split."""
        df = df.copy()
        df["question_type"] = df["task"].str.lower() + "_" + df["type"].str.lower()
        df["gt_idx"] = df["answer"].apply(lambda x: ord(x[1]) - ord("A"))
        df["gt_option"] = df.apply(lambda row: row["choices"][row["gt_idx"]], axis=1)
        df["n_options"] = df["choices"].apply(len)
        # Underlying source image (several questions can share one image); used for grouped folds.
        df["image_id"] = df["source"].astype(str) + "/" + df["source_filename"].astype(str)

        df["question_format"] = "mc"

        return df

    def check_feature_set(self, feature_set: str) -> None:
        if feature_set == "paper":
            raise ValueError(PAPER_PRESET_UNAVAILABLE)
        super().check_feature_set(feature_set)

    def get_feature_based_models(self, feature_set: str = "default") -> List[FeatureBasedBiasModel]:
        """Get per-question-type models for RandomForest evaluation."""
        self.check_feature_set(feature_set)
        return [
            Count2DModel(),
            Relation2DModel(),
            Depth3DModel(),
            Distance3DModel(),
        ]

    def get_qa_models(self) -> List[MCBenchmarkQAModel]:
        """TsT-LLM model: all CV-Bench questions are multiple choice."""
        return [MCBenchmarkQAModel(benchmark_name=self.name, default_target_col="gt_idx")]

    def get_ibp_strategy(self):
        """IBP strategy: keep at least 10 questions per question type."""
        from .debiasing import CVBIBPStrategy

        return CVBIBPStrategy()
