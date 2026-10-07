"""Regenerate the test fixtures: small stratified, text-only subsets of the benchmarks.

Each fixture holds ``N_PER_TYPE`` questions per question type, sampled with a fixed
seed from the HF test split at the pinned revision, keeping only the text columns
TsT-RF reads (no images, no videos). They are a few percent of each benchmark, not
its answer key.

    python tests/fixtures/make_fixtures.py
"""

import json
from pathlib import Path

from TsT.benchmarks.cvb.benchmark import CVB_REVISION, CVBBenchmark
from TsT.benchmarks.vsi.benchmark import VSI_REVISION, VSIBenchmark
from TsT.utils import load_hf_split

HERE = Path(__file__).parent
N_PER_TYPE = 30
SEED = 0

SPECS = {
    "vsi_test_subset.jsonl": (
        VSIBenchmark,
        VSI_REVISION,
        "question_type",
        ["id", "dataset", "scene_name", "question_type", "question", "options", "ground_truth"],
    ),
    "cvb_test_subset.jsonl": (
        CVBBenchmark,
        CVB_REVISION,
        None,  # stratify by task + type
        ["idx", "type", "task", "question", "choices", "answer", "source", "source_dataset", "source_filename"],
    ),
}


def main():
    for name, (bench_cls, revision, strat_col, cols) in SPECS.items():
        bench = bench_cls()
        df = load_hf_split(bench.hf_repo, revision, drop_columns=("image",))
        bench.check_row_order(df, revision)
        df = df[cols]
        strata = df[strat_col] if strat_col else df["task"] + "_" + df["type"]
        subset = df.groupby(strata).sample(n=N_PER_TYPE, random_state=SEED).sort_index()
        with open(HERE / name, "w") as f:
            for rec in subset[cols].to_dict(orient="records"):
                rec = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rec.items()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{name}: {len(subset)} of {len(df)} rows @ {revision[:7]}")


if __name__ == "__main__":
    main()
