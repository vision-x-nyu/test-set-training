"""MMStar benchmark: TsT-LLM only.

MMStar (Chen et al., 2024, "Are We on the Right Way for Evaluating Large Vision-Language
Models?") has 1,500 multiple-choice questions curated to require the image. The Hugging Face
`val` split embeds the answer choices in the `question` string in mixed label styles
("Options: A: x, B: y", "(A) x" or "A. x"); `preprocess` parses them into a stem and an
`options` list.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import pandas as pd

from ...core.benchmark import Benchmark, BenchmarkRegistry
from ...core.qa_models import MCBenchmarkQAModel, QuestionAnswerBiasModel
from ...utils import load_hf_split

MMSTAR_REVISION = "bc98d668301da7b14f648724866e57302778ab27"
MMSTAR_ROW_ORDER_FINGERPRINTS = {MMSTAR_REVISION: "bbe1ef42741235daf7c6ab7f7ef8224f1867f635"}

# --- Question/options parsing -------------------------------------------------
# Style-specific option-label patterns. Colon style is anchored to start-of-block
# or a preceding comma + whitespace so option text like "Springfield, D.C." is NOT
# mis-split as a "D" label (real labels look like ", D: "). Paren matches "(A)",
# dot matches "A." at a comma/line boundary (rare).
_STYLE_PATTERNS = {
    "colon": re.compile(r"(?:^|,)\s*([A-H]):\s+"),
    "paren": re.compile(r"\(([A-H])\)\s*"),
    "dot": re.compile(r"(?:^|,|\n)\s*([A-H])\.\s+"),
}
_HEADER_RE = re.compile(r"\n?\s*(?:options?|choices?)\s*:\s*", re.IGNORECASE)
_FIRST_LABEL_RE = re.compile(r"(?:^|[\s,;])(\(A\)|A:\s|A\.\s)")


def _slice_options(block: str, pat: re.Pattern) -> Optional[List[str]]:
    """Slice option texts between consecutive label matches for one style.

    Returns the options list iff the detected labels form a contiguous
    A, B, C, ... sequence of length >= 2, else None.
    """
    labels = list(pat.finditer(block))
    if len(labels) < 2:
        return None
    letters = [m.group(1) for m in labels]
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    if letters != expected:
        return None
    options: List[str] = []
    for i, lab in enumerate(labels):
        start = lab.end()
        end = labels[i + 1].start() if i + 1 < len(labels) else len(block)
        opt = block[start:end].strip()
        opt = re.sub(r"[,;]\s*$", "", opt).strip()  # drop trailing separator
        options.append(opt)
    return options


def parse_question(raw: str) -> Tuple[str, Optional[List[str]]]:
    """Split an MMStar `question` string into (stem, options_list).

    Tries all label styles and keeps the one yielding the LONGEST contiguous
    A,B,C,... option sequence (tie-break: colon > paren > dot), which is robust
    to option text that itself contains letter-like tokens. Returns
    (stem_or_raw, None) if no options can be detected.
    """
    if not raw:
        return raw, None
    text = str(raw).strip()

    m = _HEADER_RE.search(text)
    if m:
        stem = text[: m.start()].strip()
        block = text[m.end() :].strip()
    else:
        first = _FIRST_LABEL_RE.search(text)
        if not first:
            return text, None
        stem = text[: first.start(1)].strip()
        block = text[first.start(1) :].strip()

    best: Optional[List[str]] = None
    for style in ("colon", "paren", "dot"):  # priority order for tie-break
        opts = _slice_options(block, _STYLE_PATTERNS[style])
        if opts and (best is None or len(opts) > len(best)):
            best = opts
    return stem, best


@BenchmarkRegistry.register
class MMStarBenchmark(Benchmark):
    """MMStar `val` split (1,500 multiple-choice questions)."""

    name = "mmstar"
    description = "Vision-indispensable multiple-choice VQA, 1,500 curated questions (TsT-LLM only)"
    hf_repo = "Lin-Chen/MMStar"
    default_revision = MMSTAR_REVISION
    id_col = "index"
    row_order_fingerprints = MMSTAR_ROW_ORDER_FINGERPRINTS

    def load_data(self, revision: Optional[str] = None) -> pd.DataFrame:
        """Load the `val` split's text columns (the parquet file embeds images: about 42 MB)."""
        revision = self.resolve_revision(revision)
        df = load_hf_split(self.hf_repo, revision, pattern="mmstar.parquet", drop_columns=("image",))
        self.check_row_order(df, revision)
        return self.preprocess(df)

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Split each question into a stem and options; map the answer letter to gt_idx and gt_val."""
        df = df.copy()
        parsed = df["question"].apply(parse_question)
        df["question"] = parsed.apply(lambda t: t[0])
        df["options_raw"] = parsed.apply(lambda t: list(t[1]) if t[1] is not None else [])
        unparsed = df["options_raw"].apply(len) < 2
        if unparsed.any():
            raise ValueError(f"{int(unparsed.sum())} MMStar questions did not parse into two or more options")
        # Options are pre-labelled ("A. <text>"). The blind-QA converter does not re-label an option
        # whose text already starts with its own letter, which would drop the label from options
        # such as "A red kite."
        df["options"] = df["options_raw"].apply(lambda opts: [f"{chr(65 + i)}. {o}" for i, o in enumerate(opts)])
        df["num_options"] = df["options"].apply(len)

        df["ground_truth"] = df["answer"].astype(str).str.strip()
        letters = "ABCDEFGHI"
        bad = ~df["ground_truth"].isin(list(letters))
        if bad.any():
            raise ValueError(f"{int(bad.sum())} MMStar rows have a non-letter answer")
        df["gt_idx"] = df["ground_truth"].map(letters.index)
        if (df["gt_idx"] >= df["num_options"]).any():
            raise ValueError("An MMStar answer letter is beyond the question's options")
        df["gt_val"] = df.apply(lambda r: r["options_raw"][r["gt_idx"]], axis=1)
        df["question_format"] = "mc"

        # The 6 high-level categories are the question types; the 18 l2 categories are kept.
        df["category"] = df["category"].astype(str)
        df["l2_category"] = df["l2_category"].astype(str)
        df["question_type"] = df["category"]
        df["question_type_l2"] = df["l2_category"]
        return df

    def get_qa_models(self) -> List[QuestionAnswerBiasModel]:
        return [MCBenchmarkQAModel(benchmark_name=self.name)]

    def get_ibp_strategy(self):
        """IBP strategy: keep at least 5 questions per category."""
        from .debiasing import MMStarIBPStrategy

        return MMStarIBPStrategy()
