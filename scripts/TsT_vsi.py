import argparse
from TsT.benchmarks.vsi import get_models, load_data
from TsT import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        help="Comma-separated list of question types to evaluate (e.g. 'object_counting,object_abs_distance')",
    )
    parser.add_argument(
        "--target_col",
        "-t",
        type=str,
        default="ground_truth",
        help="Column to use as target variable (default: ground_truth)",
    )
    args = parser.parse_args()

    # Load data and get models
    df_full = load_data()
    models = get_models()

    # Parse question types if provided
    question_types = None
    if args.question_types is not None:
        question_types = [q.strip() for q in args.question_types.split(",")]

    run_evaluation(
        models=models,
        df_full=df_full,
        n_splits=args.n_splits,
        random_state=args.random_state,
        verbose=args.verbose,
        repeats=args.repeats,
        question_types=question_types,
        target_col=args.target_col,
    )
