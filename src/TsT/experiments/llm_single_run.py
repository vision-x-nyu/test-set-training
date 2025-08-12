"""
Single LLM experiment runner with full result capture and management.

Provides a high-level interface for running single LLM experiments with
proper logging, result capture, and metadata management.
"""

import json
import sys
import contextlib
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

from ezcolorlog import root_logger as logger
from ..evaluation import run_evaluation
from ..evaluators.llm.config import LLMRunConfig


@contextlib.contextmanager
def capture_output():
    """Capture stdout and stderr while still showing output to user."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    class TeeWriter:
        def __init__(self, original, capture):
            self.original = original
            self.capture = capture

        def write(self, text):
            # Handle both str and bytes
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            self.original.write(text)
            self.capture.write(text)
            return len(text)

        def flush(self):
            self.original.flush()
            self.capture.flush()

    try:
        sys.stdout = TeeWriter(old_stdout, stdout_capture)
        sys.stderr = TeeWriter(old_stderr, stderr_capture)
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def create_experiment_dir(base_dir: str = "results", prefix: str = "llm_single_run") -> Path:
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_experiment_config(config: LLMRunConfig, results_dir: Path) -> Path:
    """Save the LLM configuration to JSON."""
    config_dict = {
        "model_name": config.model_name,
        "max_seq_length": config.max_seq_length,
        "eval_batch_size": config.eval_batch_size,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "apply_chat_template": config.apply_chat_template,
        "learning_rate": config.learning_rate,
        "train_batch_size": config.train_batch_size,
        "num_epochs": config.num_epochs,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "template": config.template,
    }

    config_path = results_dir / "llm_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Saved LLM config to: {config_path}")
    return config_path


def save_experiment_metadata(
    results_dir: Path,
    start_time: datetime,
    end_time: datetime,
    benchmark: str,
    success: bool = True,
    error_msg: Optional[str] = None,
) -> Path:
    """Save experiment metadata."""
    metadata = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "benchmark": benchmark,
        "mode": "llm",
        "success": success,
    }

    if error_msg:
        metadata["error"] = error_msg

    metadata_path = results_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to: {metadata_path}")
    return metadata_path


def save_experiment_logs(stdout_capture: StringIO, stderr_capture: StringIO, results_dir: Path) -> Path:
    """Save captured logs."""
    logs_path = results_dir / "logs.txt"
    with open(logs_path, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout_capture.getvalue())
        f.write("\n=== STDERR ===\n")
        f.write(stderr_capture.getvalue())
    logger.info(f"Saved logs to: {logs_path}")
    return logs_path


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
    results_dir = create_experiment_dir(results_base_dir, f"llm_{benchmark}_run")
    logger.info(f"Results will be saved to: {results_dir}")

    # Save configuration
    save_experiment_config(llm_config, results_dir)

    # Import benchmark module dynamically
    try:
        import importlib

        benchmark_module = importlib.import_module(f"TsT.benchmarks.{benchmark}")
    except ImportError:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Load data and models
    logger.info(f"Loading {benchmark} data and models...")
    df_full = benchmark_module.load_data()
    models = benchmark_module.get_models()

    logger.info(f"Loaded {len(df_full)} examples from {benchmark}")
    logger.info(f"Found {len(models)} models: {[m.name for m in models]}")

    # Determine target column
    if benchmark in ["cvb", "video_mme", "mmmu"]:
        target_col = "gt_idx"
    else:
        target_col = "ground_truth"

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
        save_experiment_logs(stdout_capture, stderr_capture, results_dir)

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
            save_experiment_logs(stdout_capture, stderr_capture, results_dir)
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
    save_experiment_metadata(results_dir, start_time, end_time, benchmark, success, error_msg)

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
