"""
Video-MME benchmark: TsT-LLM only. All 2,700 questions are 4-option multiple choice.
"""

from typing import List, Optional

import pandas as pd

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.qa_models import MCBenchmarkQAModel
from ...utils import load_hf_split

# The Hub now redirects this repo id to lmms-eval/Video-MME (same commit history).
VIDEO_MME_REVISION = "ead1408f75b618502df9a1d8e0950166bf0a2a0b"
VIDEO_MME_ROW_ORDER_FINGERPRINTS = {VIDEO_MME_REVISION: "8022141d9f172087b5b10862fabbb8ff44634612"}


@BenchmarkRegistry.register
class VideoMMEBenchmark(Benchmark):
    """Video-MME test split (annotations only; no videos are downloaded)."""

    name = "video_mme"
    description = "Video understanding across domains and task types (TsT-LLM only)"
    hf_repo = "lmms-lab/Video-MME"
    default_revision = VIDEO_MME_REVISION
    id_col = "question_id"
    group_col_hint = "videoID"
    row_order_fingerprints = VIDEO_MME_ROW_ORDER_FINGERPRINTS

    def load_data(self, revision: Optional[str] = None) -> pd.DataFrame:
        revision = self.resolve_revision(revision)
        df = load_hf_split(self.hf_repo, revision, pattern="videomme/test-*.parquet")
        self.check_row_order(df, revision)
        return self.preprocess(df)

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not df["answer"].isin(list("ABCD")).all():
            raise ValueError("Video-MME answers must be one of A-D")
        df["ground_truth"] = df["answer"]
        df["gt_idx"] = df["answer"].map("ABCD".index)
        df["gt_val"] = df["answer"]
        df["question_type"] = df["task_type"]
        df["question_format"] = "mc"
        return df

    def get_qa_models(self) -> List[MCBenchmarkQAModel]:
        return [MCBenchmarkQAModel(benchmark_name=self.name, default_target_col="gt_idx")]

    def get_ibp_strategy(self):
        """IBP strategy: keep at least 5 questions per task type."""
        from .debiasing import VideoMMEIBPStrategy

        return VideoMMEIBPStrategy()
