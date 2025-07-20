import argparse
import importlib
from TsT import run_evaluation


def get_benchmark_module(benchmark_name: str):
    """Import the benchmark module dynamically."""
    try:
        module = importlib.import_module(f"TsT.benchmarks.{benchmark_name}")
        return module
    except ImportError:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. Available benchmarks: vsi, cvb"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Test-Set Training evaluation on benchmarks"
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        required=True,
        choices=["vsi", "cvb"],
        help="Benchmark to run (vsi or cvb)",
    )
    parser.add_argument(
        "--n_splits", "-k", type=int, default=5, help="Number of CV splits"
    )
    parser.add_argument(
        "--random_state", "-s", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed output"
    )
    parser.add_argument(
        "--repeats",
        "-r",
        type=int,
        default=1,
        help="Number of times to repeat evaluation with different random seeds",
    )
    parser.add_argument(
        "--question_types",
        "-q",
        type=str,
        default=None,
        help="Comma-separated list of question types to evaluate",
    )
    parser.add_argument(
        "--target_col",
        "-t",
        type=str,
        default=None,
        help="Column to use as target variable (defaults to benchmark-specific default)",
    )
    args = parser.parse_args()

    # Import the benchmark module
    benchmark_module = get_benchmark_module(args.benchmark)

    # Load data and get models
    df_full = benchmark_module.load_data()
    models = benchmark_module.get_models()

    # Set default target column if not specified
    if args.target_col is None:
        # For CVB, most models use gt_idx
        if args.benchmark == "cvb":
            target_col = "gt_idx"
        else:
            target_col = "ground_truth"  # default for VSI
    else:
        target_col = args.target_col

    # Parse question types if provided
    question_types = None
    if args.question_types is not None:
        question_types = [q.strip() for q in args.question_types.split(",")]

    print(f"Running {args.benchmark.upper()} benchmark...")
    print(f"Target column: {target_col}")
    print(f"Number of models: {len(models)}")
    if question_types:
        print(f"Question types: {question_types}")
    print()

    run_evaluation(
        models=models,
        df_full=df_full,
        n_splits=args.n_splits,
        random_state=args.random_state,
        verbose=args.verbose,
        repeats=args.repeats,
        question_types=question_types,
        target_col=target_col,
    )
