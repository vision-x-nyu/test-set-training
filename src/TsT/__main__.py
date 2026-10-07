"""
Command-line interface for the Test-set Stress-Test (TsT) diagnostic.

Run with: python -m TsT --benchmark vsi [--mode rf|llm] [options]
"""

import argparse
import logging
import re
import sys

EPILOG = """\
examples:
  python -m TsT --benchmark vsi --group_col scene_name --output_dir outputs/vsi
  python -m TsT --benchmark cvb --group_col image_id --output_dir outputs/cvb
  python -m TsT --benchmark vsi --feature_set paper   # proceedings' VSI-Bench number; reads held-out labels
  python -m TsT --benchmark vsi --mode llm --output_dir outputs/vsi_llm   # TsT-LLM; one GPU, `llm` extra

exit status: 0 on success; 1 if a question type failed (with --keep_going the others still run and
the means exclude the failed ones) or if the loaded data do not match the requested dataset revision;
2 on invalid arguments.
"""


def _short_rev(revision):
    return revision[:7] if revision and re.fullmatch(r"[0-9a-f]{40}", revision) else revision


class _LevelPrefixFormatter(logging.Formatter):
    """Plain messages for INFO; 'WARNING: ' / 'ERROR: ' prefixes otherwise (unless already present)."""

    def format(self, record):
        msg = super().format(record)
        if record.levelno >= logging.WARNING and not msg.lower().startswith(record.levelname.lower()):
            msg = f"{record.levelname}: {msg}"
        return msg


def create_parser(available_benchmarks=None):
    """Create and configure the argument parser."""
    if available_benchmarks is None:
        from .core.benchmark import BenchmarkRegistry

        available_benchmarks = BenchmarkRegistry.list_benchmarks()

    parser = argparse.ArgumentParser(
        prog="python -m TsT",
        description=(
            "Run the Test-set Stress-Test (TsT) diagnostic: k-fold cross-validation on a benchmark's test "
            "set using its non-visual inputs only, with a Random Forest on hand-crafted features (TsT-RF, "
            "CPU) or a LoRA-fine-tuned text-only LLM (TsT-LLM, one GPU)."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        required=True,
        choices=available_benchmarks,
        help=f"Benchmark to run: {', '.join(available_benchmarks)}",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="rf",
        choices=["rf", "llm"],
        help="rf: TsT-RF, a Random Forest on hand-crafted features (CPU; vsi and cvb). llm: TsT-LLM, LoRA "
        "fine-tuning of a text-only LLM (needs the llm extra and one NVIDIA GPU; all benchmarks). Default: rf.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Hugging Face dataset revision (commit sha, tag or branch). Default: the benchmark's pinned "
        "revision for the mode (vsi: bc96b17 for rf, d7cb1a3 for llm, as in the paper; cvb: bc284db). Row "
        "order changes fold assignment, so other revisions can give slightly different scores.",
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="default",
        choices=["default", "paper"],
        help="TsT-RF only. default: leak-free features (recommended). paper: VSI-Bench only; the "
        "proceedings' feature list, which includes features looked up with the held-out answer (prints a "
        "warning).",
    )
    parser.add_argument(
        "--group_col",
        type=str,
        default=None,
        help="Keep rows sharing this column's value in one fold (vsi: scene_name, cvb: image_id). "
        "Default: question-level folds.",
    )
    parser.add_argument("--n_splits", "-k", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument(
        "--repeats",
        "-r",
        type=int,
        default=1,
        help="Number of repeats with different seeds (random_state + repeat index; default: 1)",
    )
    parser.add_argument("--random_state", "-s", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--fold_seed",
        type=int,
        default=None,
        help="Seed for the fold partition only (default: --random_state). Use it to measure how much the "
        "score depends on which questions land in which fold.",
    )
    parser.add_argument(
        "--question_types",
        "-q",
        type=str,
        default=None,
        help="Comma-separated model names to evaluate, as in the 'question type' column of the results table "
        "(default: all). Dataset question types that one model scores together, such as VSI-Bench's "
        "object_rel_direction_easy/medium/hard (model: object_rel_direction), select that model. "
        "Unknown names are an error.",
    )
    parser.add_argument(
        "--target_col",
        "-t",
        type=str,
        default=None,
        help="Target column (default: each question type's own target)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Write summary.json (run settings and scores) and s_x.csv (per-question scores) here",
    )
    parser.add_argument(
        "--keep_going",
        action="store_true",
        help="If a question type fails, record it and continue with the others (exit status is still 1)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Log per-fold scores and feature importances")

    llm = parser.add_argument_group("TsT-LLM (--mode llm; defaults are the paper's settings, App. D.1)")
    llm.add_argument(
        "--llm_model", type=str, default="Qwen/Qwen2-7B-Instruct", help="Base model (default: %(default)s)"
    )
    llm.add_argument(
        "--llm_model_revision",
        type=str,
        default=None,
        help="Base-model revision on the Hugging Face Hub (default: for Qwen2-7B-Instruct, the paper's "
        "f2826a00ceef68f0f2b946d945ecc0477ce4450c; for other models, main)",
    )
    llm.add_argument("--llm_train_batch_size", type=int, default=4, help="LoRA training batch size (default: 4)")
    llm.add_argument("--llm_eval_batch_size", type=int, default=4, help="vLLM batch size (default: 4)")
    llm.add_argument("--llm_epochs", type=int, default=1, help="LoRA epochs per fold (default: 1)")
    return parser


def main(argv=None) -> int:
    """Main CLI entry point. Returns the process exit status."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Configure only the TsT loggers; third-party libraries keep their own logging setup.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_LevelPrefixFormatter("%(message)s"))
    tst_logger = logging.getLogger("TsT")
    tst_logger.handlers = [handler]
    tst_logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
    tst_logger.propagate = False

    from .core.benchmark import BenchmarkRegistry, DatasetRevisionError
    from .evaluation import LOCKED_VERSIONS, ModelEvaluationError, evaluate_benchmark, format_results_table
    from .evaluation import resolve_question_types
    from .evaluation import version_mismatches

    if args.n_splits < 2 or args.repeats < 1:
        print("error: --n_splits must be at least 2 and --repeats at least 1.", file=sys.stderr)
        return 2
    benchmark = BenchmarkRegistry.get_benchmark(args.benchmark)
    if args.mode == "rf" and not benchmark.supports_rf:
        print(f"error: TsT-RF is not available for benchmark '{benchmark.name}'; use --mode llm.", file=sys.stderr)
        return 2
    if args.mode == "llm" and args.feature_set != "default":
        print("error: --feature_set applies to TsT-RF only (--mode rf).", file=sys.stderr)
        return 2
    if args.mode == "rf":
        try:
            benchmark.check_feature_set(args.feature_set)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    question_types = None
    if args.question_types is not None:
        try:
            # model names do not depend on the feature set
            models = benchmark.get_feature_based_models() if args.mode == "rf" else benchmark.get_qa_models()
            question_types = resolve_question_types(models, args.question_types.split(","))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    llm_config = None
    if args.mode == "llm":
        from .evaluators.llm import LLMRunConfig, missing_requirements

        problem = missing_requirements()
        if problem:
            print(f"error: {problem}.", file=sys.stderr)
            return 2

        llm_config = LLMRunConfig(
            model_name=args.llm_model,
            model_revision=args.llm_model_revision,
            train_batch_size=args.llm_train_batch_size,
            eval_batch_size=args.llm_eval_batch_size,
            num_epochs=args.llm_epochs,
        )

    mismatches = version_mismatches()
    if mismatches:
        got = ", ".join(f"{pkg} {v['installed']} (locked: {v['locked']})" for pkg, v in mismatches.items())
        print(
            f"WARNING: {got}. The published values were computed with "
            + ", ".join(f"{p} {v}" for p, v in LOCKED_VERSIONS.items())
            + "; other versions can move the headline score by about a point and per-type scores by "
            "several points. Install with `uv sync --frozen` or `pip install -c constraints.txt .` to reproduce them.",
            file=sys.stderr,
        )

    revision = benchmark.resolve_revision(args.revision, mode=args.mode)
    folds = f"grouped by {args.group_col}" if args.group_col else "question-level"
    method = f"feature set: {args.feature_set}" if args.mode == "rf" else f"model: {args.llm_model}"
    print(
        f"TsT-{args.mode.upper()} | benchmark: {benchmark.name} ({benchmark.hf_repo} @ {_short_rev(revision)}) | "
        f"{method} | {args.n_splits}-fold, {folds} | repeats: {args.repeats} | "
        f"seed: {args.random_state}" + (f" | fold seed: {args.fold_seed}" if args.fold_seed is not None else ""),
        flush=True,
    )

    try:
        run = evaluate_benchmark(
            benchmark,
            mode=args.mode,
            revision=revision,
            feature_set=args.feature_set,
            group_col=args.group_col,
            n_splits=args.n_splits,
            repeats=args.repeats,
            random_state=args.random_state,
            fold_seed=args.fold_seed,
            question_types=question_types,
            target_col=args.target_col,
            keep_going=args.keep_going,
            verbose=args.verbose,
            llm_config=llm_config,
            predictions_dir=(f"{args.output_dir}/predictions" if args.output_dir and args.mode == "llm" else None),
        )
    except DatasetRevisionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ModelEvaluationError as e:
        if args.verbose:
            import traceback

            traceback.print_exc()
        print(f"error: {e}" + ("" if args.verbose else " (rerun with --verbose for the traceback)"), file=sys.stderr)
        return 1

    print()
    print(format_results_table(run.results))
    if args.output_dir:
        paths = run.save(args.output_dir)
        print(f"wrote {paths['summary']} and {paths['s_x']}")

    if run.errors:
        print(
            f"ERROR: {len(run.errors)} question type(s) failed: {', '.join(run.errors)}. The means above exclude them.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
