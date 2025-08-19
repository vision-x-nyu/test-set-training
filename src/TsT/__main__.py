"""
Command-line interface for Test-Set Training (TsT) evaluation.

This module provides a CLI for running TsT evaluations on various benchmarks.
Run with: python -m TsT [options]
"""

import argparse
import importlib

from ezcolorlog import root_logger as logger
from .evaluation import run_evaluation
from .evaluators.llm.config import LLMRunConfig


def get_benchmark_module(benchmark_name: str):
    """Import the benchmark module dynamically."""
    try:
        module = importlib.import_module(f"TsT.benchmarks.{benchmark_name}")
        return module
    except ImportError:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available benchmarks: vsi, cvb, video_mme, mmmu")


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Run Test-Set Training evaluation on benchmarks")
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        required=True,
        choices=["vsi", "cvb", "video_mme", "mmmu"],
        help="Benchmark to run (vsi, cvb, video_mme, or mmmu)",
    )
    parser.add_argument("--n_splits", "-k", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--random_state", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed output")
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
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["rf", "llm"],
        default="rf",
        help="Evaluation mode: 'rf' for Random Forest, 'llm' for LLM-based TsT",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="google/gemma-2-2b-it",
        help="LLM model to use for LLM mode (default: google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--llm_train_batch_size",
        type=int,
        default=4,
        help="Batch size for LLM training (default: 4)",
    )
    parser.add_argument(
        "--llm_eval_batch_size",
        type=int,
        default=4,
        help="Batch size for LLM evaluation (default: 4)",
    )
    parser.add_argument(
        "--llm_epochs",
        type=int,
        default=1,
        help="Number of training epochs for LLM fine-tuning (default: 1)",
    )
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Import the benchmark module
    benchmark_module = get_benchmark_module(args.benchmark)

    # Load data and get models
    df_full = benchmark_module.load_data()
    models = benchmark_module.get_models()

    # Set default target column if not specified
    if args.target_col is None:
        # For CVB and Video-MME, most models use gt_idx
        if args.benchmark in ["cvb", "video_mme"]:
            target_col = "gt_idx"
        else:
            target_col = "ground_truth"  # default for VSI
    else:
        target_col = args.target_col

    # Parse question types if provided
    question_types = None
    if args.question_types is not None:
        question_types = [q.strip() for q in args.question_types.split(",")]

    # Create LLM config if using LLM mode
    llm_config = None
    if args.mode == "llm":
        llm_config = LLMRunConfig(
            model_name=args.llm_model,
            train_batch_size=args.llm_train_batch_size,
            eval_batch_size=args.llm_eval_batch_size,
            learning_rate=2e-4,
            num_epochs=args.llm_epochs,
            lora_rank=8,
            lora_alpha=16,
            max_seq_length=1024,
        )

    logger.info(f"Running {args.benchmark.upper()} benchmark...")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Target column: {target_col}")
    logger.info(f"Number of models: {len(models)}")
    if question_types:
        logger.info(f"Question types: {question_types}")
    if args.mode == "llm":
        logger.info(f"LLM model: {args.llm_model}")
        logger.info(f"LLM train batch size: {args.llm_train_batch_size}")
        logger.info(f"LLM eval batch size: {args.llm_eval_batch_size}")
        logger.info(f"LLM epochs: {args.llm_epochs}")
    logger.info("")

    run_evaluation(
        question_models=models,
        df_full=df_full,
        n_splits=args.n_splits,
        random_state=args.random_state,
        verbose=args.verbose,
        repeats=args.repeats,
        question_types=question_types,
        target_col=target_col,
        mode=args.mode,
        llm_config=llm_config,
    )


if __name__ == "__main__":
    main()
