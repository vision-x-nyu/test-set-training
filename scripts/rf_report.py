#!/usr/bin/env python3
"""TsT-RF report for one benchmark: per-question-type and overall scores as JSON.

Runs the shipped feature-based RF models (the same run as ``python -m TsT``) for a
named feature set, with question-level (default) or grouped folds, on a pinned HF
dataset revision, and records a row-order fingerprint (shuffled k-fold assignment
depends on row order). Prints the count-weighted and macro means to 2 decimals.

Examples:
    python scripts/rf_report.py --benchmark vsi --out vsi_default_random.json
    python scripts/rf_report.py --benchmark vsi --group_col scene_name --out vsi_default_scene.json
    python scripts/rf_report.py --benchmark vsi --feature_set paper --out vsi_paper_random.json
    python scripts/rf_report.py --benchmark cvb --group_col image_id --out cvb_default_image.json
"""

import argparse
import json
import sys

from TsT.core.benchmark import BenchmarkRegistry, DatasetRevisionError
from TsT.evaluation import evaluate_benchmark


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=["vsi", "cvb"])
    ap.add_argument("--revision", default=None, help="HF dataset revision (default: the benchmark's pinned revision)")
    ap.add_argument(
        "--feature_set",
        default="default",
        choices=["default", "paper"],
        help="default: leak-free (recommended); paper: VSI-Bench only, the proceedings' feature list",
    )
    ap.add_argument("--group_col", default=None, help="Grouped folds by this column (default: question-level)")
    ap.add_argument("--target_col", default=None, help="Default: each question type's own target")
    ap.add_argument("--drop_features", default="", help="Comma-separated feature columns to drop from every model")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--keep_going", action="store_true", help="Record failed question types and continue")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args(argv)

    bench = BenchmarkRegistry.get_benchmark(args.benchmark)
    try:
        bench.check_feature_set(args.feature_set)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        run = evaluate_benchmark(
            bench,
            revision=args.revision,
            feature_set=args.feature_set,
            group_col=args.group_col,
            n_splits=args.n_splits,
            repeats=args.repeats,
            random_state=args.random_state,
            target_col=args.target_col,
            keep_going=args.keep_going,
            show_progress=False,
            drop_features=[c.strip() for c in args.drop_features.split(",") if c.strip()],
        )
    except DatasetRevisionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = run.summary()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    def pct(x):
        return "n/a" if x is None else f"{100 * x:.2f}"

    if out["environment"]["differs_from_lock"]:
        print(
            f"WARNING: versions differ from the lock, so values can differ from the published ones: "
            f"{out['environment']['differs_from_lock']}",
            file=sys.stderr,
        )

    print(
        f"[{run.benchmark} rev={run.revision[:7]} features={run.feature_set} group={run.group_col} "
        f"drop={run.dropped_features}] weighted={pct(out['weighted_mean'])} macro={pct(out['macro_mean'])} "
        f"-> {args.out}"
    )
    for p in out["per_question_type"]:
        score = "FAILED" if p["error"] else pct(p["score"])
        print(f"   {p['question_type']:24s} n={p['count']:5d} {p['metric']}={score}")
    if out["errors"]:
        print(f"ERRORS: {out['errors']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
