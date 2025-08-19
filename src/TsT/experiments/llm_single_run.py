"""
Single LLM experiment runner with full result capture and management.

Provides a high-level interface for running single LLM experiments with
proper logging, result capture, and metadata management.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from ezcolorlog import root_logger as logger
from ..evaluation import run_evaluation
from ..evaluators.llm.config import LLMRunConfig
from .utils import (
    capture_output,
    create_timestamped_dir,
    save_llm_config,
    save_metadata,
    save_logs,
    load_benchmark,
    get_target_column,
)


def run_single_llm_experiment(
    llm_config: LLMRunConfig,
    benchmark: str,
    results_base_dir: str = "results",
    n_splits: int = 3,
    random_state: int = 42,
    question_types: Optional[list] = None,
    verbose: bool = True,
) -> Path:
    """
    Run a single LLM experiment with full result capture.

    Args:
        llm_config: LLM configuration with hyperparameters
        benchmark: Benchmark name (e.g., "mmmu", "vsi", "cvb")
        results_base_dir: Base directory for results
        n_splits: Number of CV splits
        random_state: Random seed
        question_types: Optional list of question types to evaluate
        verbose: Whether to print detailed output

    Returns:
        Path to the results directory
    """
    print("=" * 80)
    print(f"LLM SINGLE RUN - {benchmark.upper()} EVALUATION")
    print("=" * 80)

    # Create results directory
    results_dir = create_timestamped_dir(results_base_dir, f"llm_{benchmark}_run")
    logger.info(f"Results will be saved to: {results_dir}")

    # Save configuration
    save_llm_config(llm_config, results_dir)

    # Load benchmark using registry system
    benchmark_obj = load_benchmark(benchmark)

    # Load data and get QA model for LLM evaluation
    logger.info(f"Loading {benchmark} data and models...")
    df_full = benchmark_obj.load_data()
    models = benchmark_obj.get_qa_models()  # Use QA models for LLM evaluation

    logger.info(f"Loaded {len(df_full)} examples from {benchmark}")
    logger.info(f"Found {len(models)} models: {[m.name for m in models]}")

    # Get target column
    target_col = get_target_column(benchmark)

    # Run evaluation with log capture
    start_time = datetime.now()
    logger.info(f"Starting evaluation at: {start_time}")

    success = True
    error_msg = None
    summary = None

    try:
        with capture_output() as (stdout_capture, stderr_capture):
            summary = run_evaluation(
                question_models=models,
                df_full=df_full,
                n_splits=n_splits,
                random_state=random_state,
                verbose=verbose,
                repeats=1,
                question_types=question_types,
                target_col=target_col,
                mode="llm",
                llm_config=llm_config,
            )

        end_time = datetime.now()
        logger.info(f"Evaluation completed at: {end_time}")

        # Save captured logs
        save_logs(stdout_capture, stderr_capture, results_dir)

        # Save results
        if summary is not None:
            results_path = results_dir / "results.csv"
            summary.to_csv(results_path, index=False)
            logger.info(f"Saved results to: {results_path}")

    except Exception as e:
        end_time = datetime.now()
        success = False
        error_msg = str(e)
        logger.error(f"Evaluation failed: {e}", exc_info=True)

        # Still save logs and error info
        try:
            save_logs(stdout_capture, stderr_capture, results_dir)
        except Exception as e:
            logger.error(f"Failed to save logs: {e}", exc_info=True)
            pass  # If log capture failed too, don't crash

        # Save error info
        error_path = results_dir / "error.txt"
        with open(error_path, "w") as f:
            f.write(f"Error occurred at: {end_time}\n")
            f.write(f"Error message: {str(e)}\n")
            f.write(f"Duration before error: {(end_time - start_time).total_seconds():.1f} seconds\n")

    # Save metadata
    metadata = {
        "benchmark": benchmark,
        "mode": "llm",
        "success": success,
    }
    if error_msg:
        metadata["error"] = error_msg

    save_metadata(results_dir, start_time, end_time, metadata)

    # Print summary
    duration = (end_time - start_time).total_seconds()
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    if success and summary is not None:
        print(summary.to_string(index=False))
    else:
        print(f"❌ Experiment failed: {error_msg}")

    print(f"\nTotal duration: {duration:.1f} seconds")
    print(f"Results saved to: {results_dir}")

    return results_dir
