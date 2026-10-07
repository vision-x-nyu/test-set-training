"""
MMMU (Massive Multi-discipline Multimodal Understanding) benchmark: TsT-LLM only.

TsT-LLM diagnoses the 847 multiple-choice questions of the validation split; the 53
open-ended questions are loaded but not scored.
"""

import ast
from typing import List, Optional

import pandas as pd

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.protocols import QAFormat
from ...core.qa_models import GlobalBenchmarkQAModel, MCBenchmarkQAModel
from ...utils import load_hf_split

# The Hub now redirects this repo id to lmms-lab-encoder/MMMU (same commit history).
MMMU_REVISION = "364f2e2eb107b36e07ff4c5a15f5947a759cef47"
MMMU_ROW_ORDER_FINGERPRINTS = {MMMU_REVISION: "2f89fd1e941ac409e0e6fc83dee018709e8d40e5"}
MMMU_IMAGE_COLUMNS = tuple(f"image_{i}" for i in range(1, 8))
LETTERS = "ABCDEFGHI"


@BenchmarkRegistry.register
class MMMUBenchmark(Benchmark):
    """MMMU validation split (900 questions; TsT-LLM scores the 847 multiple-choice ones)."""

    name = "mmmu"
    description = "Multimodal understanding across academic disciplines (TsT-LLM only)"
    hf_repo = "lmms-lab/MMMU"
    default_revision = MMMU_REVISION
    id_col = "id"
    row_order_fingerprints = MMMU_ROW_ORDER_FINGERPRINTS

    def load_data(self, revision: Optional[str] = None) -> pd.DataFrame:
        """Load the validation split's text columns (the parquet file embeds images: about 340 MB)."""
        revision = self.resolve_revision(revision)
        df = load_hf_split(self.hf_repo, revision, pattern="data/validation-*.parquet", drop_columns=MMMU_IMAGE_COLUMNS)
        self.check_row_order(df, revision)
        return self.preprocess(df)

    @staticmethod
    def _question_format(mmmu_question_type: str) -> QAFormat:
        formats = {"multiple-choice": "mc", "open": "oe"}
        if mmmu_question_type not in formats:
            raise ValueError(f"Unknown MMMU question type: {mmmu_question_type!r}")
        return formats[mmmu_question_type]

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Parse options and answers; use the subfield (merged below 5 questions) as the question type."""
        df = df.rename(columns={"answer": "ground_truth"})
        df["options"] = df["options"].apply(ast.literal_eval)
        df["num_options"] = df["options"].apply(len)
        df["question_format"] = df["question_type"].apply(MMMUBenchmark._question_format)

        mc = df["question_format"] == "mc"
        bad = mc & ~df["ground_truth"].isin(list(LETTERS))
        if bad.any():
            raise ValueError(f"{int(bad.sum())} MMMU multiple-choice rows have a non-letter answer")
        df["gt_idx"] = -1
        df.loc[mc, "gt_idx"] = df.loc[mc, "ground_truth"].map(LETTERS.index)
        if (df.loc[mc, "gt_idx"] >= df.loc[mc, "num_options"]).any():
            raise ValueError("An MMMU multiple-choice answer letter is beyond the question's options")
        df["gt_val"] = None
        df.loc[mc, "gt_val"] = df[mc].apply(lambda r: r["options"][r["gt_idx"]], axis=1)

        # Subfields with fewer than 5 questions (counted over all 900, open-ended included) become
        # "Other". This grouping is the IBP diversity constraint used in the paper.
        df["subfield_truncated"] = (
            df.groupby("subfield")["subfield"].transform(lambda x: "Other" if len(x) < 5 else x).fillna("Other")
        )
        df["question_type_original"] = df["question_type"]
        df["question_type"] = df["subfield_truncated"]
        return df

    def get_qa_models(self) -> List[GlobalBenchmarkQAModel]:
        """TsT-LLM model: the multiple-choice questions (open-ended questions are not scored)."""
        return [MCBenchmarkQAModel(benchmark_name=self.name)]

    def get_ibp_strategy(self):
        """IBP strategy: keep at least 5 questions per (merged) subfield."""
        from .debiasing import MMMUIBPStrategy

        return MMMUIBPStrategy()
