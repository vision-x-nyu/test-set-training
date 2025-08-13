"""
LLM hyperparameter sweep runner with result aggregation.

Provides functionality to run multiple LLM experiments with different
hyperparameter configurations and aggregate results across runs.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ezcolorlog import root_logger as logger
from ..evaluation import run_evaluation
from ..evaluators.llm.config import LLMRunConfig
from .utils import (
    capture_output,
    save_llm_config,
    save_metadata,
    save_logs,
    save_json,
    load_benchmark_module,
    get_target_column,
    create_run_name,
    generate_experiment_name,
    validate_sweep_config,
    find_completed_runs,
)


def run_llm_sweep(
    configs: List[LLMRunConfig],
    benchmark: str,
    experiment_name: Optional[str] = None,
    results_base_dir: str = "results",
    n_splits: int = 3,
    random_state: int = 42,
    question_types: Optional[list] = None,
    verbose: bool = True,
    continue_on_failure: bool = True,
    resume: bool = True,
    force_overwrite: bool = False,
) -> Path:
    """
    Run hyperparameter sweep with multiple LLM configurations.

    Args:
        configs: List of LLMRunConfig objects with different hyperparameters
        benchmark: Benchmark name (e.g., "mmmu", "vsi", "cvb")
        experiment_name: Optional experiment name. If None, auto-generated from configs
        results_base_dir: Base directory for results
        n_splits: Number of CV splits
        random_state: Random seed
        question_types: Optional list of question types to evaluate
        verbose: Whether to print detailed output
        continue_on_failure: Whether to continue sweep if one config fails
        resume: Whether to resume existing experiments (default: True)
        force_overwrite: Whether to overwrite existing experiments (default: False)

    Returns:
        Path to the sweep results directory
    """
    print("=" * 80)
    print(f"LLM HYPERPARAMETER SWEEP - {benchmark.upper()}")
    print("=" * 80)
    print(f"Running {len(configs)} configurations...")

    # Generate or use provided experiment name
    if experiment_name is None:
        experiment_name = generate_experiment_name(configs, benchmark)
        logger.info(f"Auto-generated experiment name: {experiment_name}")
    else:
        logger.info(f"Using provided experiment name: {experiment_name}")

    # Set up experiment directory (nested under experiments/)
    sweep_dir = Path(results_base_dir) / "experiments" / experiment_name

    # Handle existing experiments
    if sweep_dir.exists():
        if not resume and not force_overwrite:
            raise ValueError(
                f"Experiment '{experiment_name}' already exists. "
                "Use resume=True to continue or force_overwrite=True to restart."
            )

        if force_overwrite:
            logger.warning(f"Force overwriting existing experiment: {sweep_dir}")
            import shutil

            shutil.rmtree(sweep_dir)
            sweep_dir.mkdir(parents=True)
        elif resume:
            logger.info(f"Resuming existing experiment: {sweep_dir}")
            # Validate existing config
            existing_config_path = sweep_dir / "sweep_config.json"
            if existing_config_path.exists():
                with open(existing_config_path, "r") as f:
                    existing_config = json.load(f)

                new_config = {
                    "benchmark": benchmark,
                    "n_configs": len(configs),
                    "configs": [_config_to_dict(config) for config in configs],
                }

                if not validate_sweep_config(existing_config, new_config):
                    raise ValueError(
                        f"Configuration mismatch for experiment '{experiment_name}'. "
                        "Configs don't match existing experiment. Use a different experiment_name "
                        "or force_overwrite=True to restart with new config."
                    )
                logger.info("✅ Configuration validated - matches existing experiment")
            else:
                logger.warning("No existing config found - treating as new experiment")
    else:
        logger.info(f"Creating new experiment: {sweep_dir}")
        sweep_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep configuration
    sweep_config = {
        "benchmark": benchmark,
        "n_configs": len(configs),
        "n_splits": n_splits,
        "random_state": random_state,
        "question_types": question_types,
        "continue_on_failure": continue_on_failure,
        "experiment_name": experiment_name,
        "resume": resume,
        "configs": [_config_to_dict(config) for config in configs],
    }
    save_json(sweep_config, sweep_dir, "sweep_config.json")

    # Check for completed runs if resuming
    completed_runs = []
    if resume and sweep_dir.exists():
        completed_runs = find_completed_runs(sweep_dir)
        if completed_runs:
            logger.info(f"Found {len(completed_runs)}/{len(configs)} completed runs:")
            for run_name in completed_runs:
                logger.info(f"  ✅ {run_name}")
        else:
            logger.info("No completed runs found - starting from beginning")

    logger.info(f"Sweep results will be saved to: {sweep_dir}")

    # Load benchmark module and data once
    benchmark_module = load_benchmark_module(benchmark)
    logger.info(f"Loading {benchmark} data and models...")
    df_full = benchmark_module.load_data()
    models = benchmark_module.get_models()
    target_col = get_target_column(benchmark)

    logger.info(f"Loaded {len(df_full)} examples from {benchmark}")
    logger.info(f"Found {len(models)} models: {[m.name for m in models]}")

    # Run each configuration (skip completed ones)
    start_time = datetime.now()
    run_results = []
    successful_runs = 0
    failed_runs = 0
    skipped_runs = 0

    for i, config in enumerate(configs, 1):
        run_name = create_run_name(config, i)
        run_dir = sweep_dir / run_name

        # Check if this run is already completed
        if resume and run_name in completed_runs:
            print(f"\n{'=' * 60}")
            print(f"RUN {i}/{len(configs)}: {run_name} [SKIPPING - ALREADY COMPLETED]")
            print(f"{'=' * 60}")

            # Load existing results
            try:
                existing_results_path = run_dir / "results.csv"
                if existing_results_path.exists():
                    summary = pd.read_csv(existing_results_path)
                    run_start = run_end = datetime.now()  # Dummy times for completed runs

                    run_result = _create_run_result(config, i, run_name, summary, run_start, run_end, success=True)
                    run_results.append(run_result)
                    successful_runs += 1
                    skipped_runs += 1

                    logger.info(f"✅ Loaded existing results for run {i}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to load existing results for {run_name}: {e}")
                # Fall through to re-run this configuration

        # Create run directory for new runs
        run_dir.mkdir(exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"RUN {i}/{len(configs)}: {run_name}")
        print(f"{'=' * 60}")

        try:
            # Run single configuration
            run_start = datetime.now()
            summary = _run_single_config(
                config, benchmark, df_full, models, target_col, run_dir, n_splits, random_state, question_types, verbose
            )
            run_end = datetime.now()

            # Record successful run
            run_result = _create_run_result(config, i, run_name, summary, run_start, run_end, success=True)
            run_results.append(run_result)
            successful_runs += 1

            logger.info(f"✅ Run {i} completed successfully")

        except Exception as e:
            run_end = datetime.now()
            error_msg = str(e)

            logger.error(f"❌ Run {i} failed: {e}", exc_info=True)

            # Save error info
            error_path = run_dir / "error.txt"
            with open(error_path, "w") as f:
                f.write(f"Run {i} failed at: {run_end}\n")
                f.write(f"Error: {error_msg}\n")

            # Record failed run
            run_result = _create_run_result(
                config, i, run_name, None, run_start, run_end, success=False, error=error_msg
            )
            run_results.append(run_result)
            failed_runs += 1

            if not continue_on_failure:
                logger.error("Stopping sweep due to failure (continue_on_failure=False)")
                break

    end_time = datetime.now()

    # Create sweep summary
    summary_df = _create_sweep_summary(run_results)
    summary_path = sweep_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved sweep summary to: {summary_path}")

    # Save sweep metadata
    sweep_metadata = {
        "benchmark": benchmark,
        "mode": "llm_sweep",
        "total_configs": len(configs),
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "skipped_runs": skipped_runs,
        "new_runs": len(configs) - skipped_runs,
        "success_rate": successful_runs / len(configs) if configs else 0,
        "experiment_name": experiment_name,
        "resumed": resume and skipped_runs > 0,
    }
    save_metadata(sweep_dir, start_time, end_time, sweep_metadata, "sweep_metadata.json")

    # Print final summary
    _print_sweep_summary(
        summary_df, successful_runs, failed_runs, skipped_runs, (end_time - start_time).total_seconds(), sweep_dir
    )

    return sweep_dir


def _run_single_config(
    config: LLMRunConfig,
    benchmark: str,
    df_full: pd.DataFrame,
    models: list,
    target_col: str,
    run_dir: Path,
    n_splits: int,
    random_state: int,
    question_types: Optional[list],
    verbose: bool,
) -> pd.DataFrame:
    """Run evaluation for a single configuration."""
    # Save config for this run
    save_llm_config(config, run_dir)

    # Run evaluation with log capture
    run_start = datetime.now()

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
            llm_config=config,
        )

    run_end = datetime.now()

    # Save logs and results
    save_logs(stdout_capture, stderr_capture, run_dir)

    if summary is not None:
        results_path = run_dir / "results.csv"
        summary.to_csv(results_path, index=False)

    # Save run metadata
    run_metadata = {
        "benchmark": benchmark,
        "mode": "llm",
        "success": True,
    }
    save_metadata(run_dir, run_start, run_end, run_metadata)

    return summary


def _config_to_dict(config: LLMRunConfig) -> Dict[str, Any]:
    """Convert LLMRunConfig to dictionary."""
    return {
        "model_name": config.model_name,
        "learning_rate": config.learning_rate,
        "train_batch_size": config.train_batch_size,
        "num_epochs": config.num_epochs,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "max_seq_length": config.max_seq_length,
        "eval_batch_size": config.eval_batch_size,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }


def _create_run_result(
    config: LLMRunConfig,
    run_id: int,
    run_name: str,
    summary: Optional[pd.DataFrame],
    start_time: datetime,
    end_time: datetime,
    success: bool,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Create result dictionary for a single run."""
    result = {
        "run_id": run_id,
        "run_name": run_name,
        "success": success,
        "duration_seconds": (end_time - start_time).total_seconds(),
        "learning_rate": config.learning_rate,
        "train_batch_size": config.train_batch_size,
        "num_epochs": config.num_epochs,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "model_name": config.model_name,
    }

    if success and summary is not None and not summary.empty:
        # Extract score from summary (assuming single model)
        score_str = summary.iloc[0]["Score"]  # e.g., "25.0%"
        score = float(score_str.rstrip("%")) / 100  # Convert to decimal
        result["score"] = score
        result["score_display"] = score_str
    else:
        result["score"] = 0.0
        result["score_display"] = "0.0%" if success else "FAILED"

    if error:
        result["error"] = error

    return result


def _create_sweep_summary(run_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create summary DataFrame from run results."""
    summary_df = pd.DataFrame(run_results)

    # Sort by score (best first)
    summary_df = summary_df.sort_values("score", ascending=False)

    return summary_df


def _print_sweep_summary(
    summary_df: pd.DataFrame,
    successful_runs: int,
    failed_runs: int,
    skipped_runs: int,
    total_duration: float,
    sweep_dir: Path,
):
    """Print formatted sweep summary."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("=" * 80)

    if not summary_df.empty:
        # Show top 5 configurations
        display_cols = ["run_name", "score_display", "learning_rate", "train_batch_size", "lora_rank"]
        top_runs = summary_df[display_cols].head(5)
        print("TOP 5 CONFIGURATIONS:")
        print(top_runs.to_string(index=False))

        # Show best configuration details
        best_run = summary_df.iloc[0]
        print("\n🏆 BEST CONFIGURATION:")
        print(f"   Run: {best_run['run_name']}")
        print(f"   Score: {best_run['score_display']}")
        print(f"   Learning Rate: {best_run['learning_rate']}")
        print(f"   Batch Size: {best_run['train_batch_size']}")
        print(f"   LoRA Rank: {best_run['lora_rank']}")

    print("\n📊 OVERALL STATS:")
    print(f"   Successful runs: {successful_runs}")
    print(f"   Failed runs: {failed_runs}")
    if skipped_runs > 0:
        print(f"   Skipped runs (resume): {skipped_runs}")
        print(f"   New runs executed: {successful_runs + failed_runs - skipped_runs}")
    total_runs = successful_runs + failed_runs
    print(f"   Success rate: {successful_runs / total_runs * 100:.1f}%" if total_runs > 0 else "   Success rate: N/A")
    print(f"   Total duration: {total_duration:.1f} seconds")
    print(f"   Results saved to: {sweep_dir}")
    print("=" * 80)
