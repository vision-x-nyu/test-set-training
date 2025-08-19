"""
LLM hyperparameter sweep runner with result aggregation.

Provides functionality to run multiple LLM experiments with different
hyperparameter configurations and aggregate results across runs.
Supports both single-benchmark and multi-benchmark sweeps.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ezcolorlog import root_logger as logger
from ..evaluation import run_evaluation
from ..evaluators.llm.config import LLMRunConfig
from .utils import (
    capture_output,
    save_llm_config,
    save_metadata,
    save_logs,
    save_json,
    load_benchmark,
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

    # Load benchmark and data once using new system
    benchmark_obj = load_benchmark(benchmark)
    logger.info(f"Loading {benchmark} data and models...")
    df_full = benchmark_obj.load_data()
    models = benchmark_obj.get_qa_models()  # Use QA model for LLM evaluation
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


def run_llm_multi_benchmark_sweep(
    configs: List[LLMRunConfig],
    benchmarks: List[str],
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
    Run hyperparameter sweep across multiple benchmarks.

    Directory structure:
    results/experiments/{experiment_name}/
    ├── sweep_config.json
    ├── sweep_summary.csv
    ├── benchmark_summary.csv      # Per-benchmark max performance
    ├── config_summary.csv         # Per-config cross-benchmark stats
    └── {benchmark_name}/
        └── run_{id}_{hyperparams}/
            ├── results.csv
            ├── metadata.json
            └── ...

    Args:
        configs: List of LLMRunConfig objects with different hyperparameters
        benchmarks: List of benchmark names (e.g., ["mmmu", "vsi", "cvb"])
        experiment_name: Optional experiment name. If None, auto-generated
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
    print(f"LLM MULTI-BENCHMARK SWEEP - {len(benchmarks)} BENCHMARKS")
    print("=" * 80)
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Running {len(configs)} configurations across {len(benchmarks)} benchmarks...")
    print(f"Total runs: {len(configs)} × {len(benchmarks)} = {len(configs) * len(benchmarks)}")

    # Generate experiment name
    if experiment_name is None:
        experiment_name = _generate_multi_benchmark_experiment_name(configs, benchmarks)
        logger.info(f"Auto-generated experiment name: {experiment_name}")
    else:
        logger.info(f"Using provided experiment name: {experiment_name}")

    # Set up experiment directory
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
                    "benchmarks": benchmarks,
                    "n_configs": len(configs),
                    "configs": [_config_to_dict(config) for config in configs],
                }

                if not _validate_multi_benchmark_config(existing_config, new_config):
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
        "benchmarks": benchmarks,
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
        completed_runs = _find_completed_multi_benchmark_runs(sweep_dir, benchmarks)
        if completed_runs:
            logger.info(f"Found {len(completed_runs)} completed (config, benchmark) pairs:")
            for config_name, benchmark in completed_runs:
                logger.info(f"  ✅ {config_name} on {benchmark}")
        else:
            logger.info("No completed runs found - starting from beginning")

    logger.info(f"Sweep results will be saved to: {sweep_dir}")

    # Run all (config, benchmark) combinations
    start_time = datetime.now()
    all_run_results = []
    successful_runs = 0
    failed_runs = 0
    skipped_runs = 0

    for config_idx, config in enumerate(configs, 1):
        run_name = create_run_name(config, config_idx)

        for benchmark in benchmarks:
            # Check if this (config, benchmark) pair is already completed
            if resume and (run_name, benchmark) in completed_runs:
                print(f"\n{'=' * 60}")
                print(
                    f"RUN {config_idx}/{len(configs)} - {benchmark.upper()}: {run_name} [SKIPPING - ALREADY COMPLETED]"
                )
                print(f"{'=' * 60}")

                # Load existing results
                try:
                    benchmark_dir = sweep_dir / benchmark
                    run_dir = benchmark_dir / run_name
                    existing_results_path = run_dir / "results.csv"
                    if existing_results_path.exists():
                        summary = pd.read_csv(existing_results_path)
                        run_start = run_end = datetime.now()  # Dummy times

                        run_result = _create_multi_benchmark_run_result(
                            config, config_idx, run_name, benchmark, summary, run_start, run_end, success=True
                        )
                        all_run_results.append(run_result)
                        successful_runs += 1
                        skipped_runs += 1

                        logger.info(f"✅ Loaded existing results for {run_name} on {benchmark}")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to load existing results for {run_name} on {benchmark}: {e}")
                    # Fall through to re-run this configuration

            # Create benchmark and run directories
            benchmark_dir = sweep_dir / benchmark
            benchmark_dir.mkdir(exist_ok=True)
            run_dir = benchmark_dir / run_name
            run_dir.mkdir(exist_ok=True)

            print(f"\n{'=' * 60}")
            print(f"RUN {config_idx}/{len(configs)} - {benchmark.upper()}: {run_name}")
            print(f"{'=' * 60}")

            try:
                # Run single (config, benchmark) combination
                run_start = datetime.now()
                summary = _run_single_config_benchmark(
                    config, benchmark, run_dir, n_splits, random_state, question_types, verbose
                )
                run_end = datetime.now()

                # Record successful run
                run_result = _create_multi_benchmark_run_result(
                    config, config_idx, run_name, benchmark, summary, run_start, run_end, success=True
                )
                all_run_results.append(run_result)
                successful_runs += 1

                logger.info(f"✅ Run {config_idx} on {benchmark} completed successfully")

            except Exception as e:
                run_end = datetime.now()
                error_msg = str(e)

                logger.error(f"❌ Run {config_idx} on {benchmark} failed: {e}", exc_info=True)

                # Save error info
                error_path = run_dir / "error.txt"
                with open(error_path, "w") as f:
                    f.write(f"Run {config_idx} on {benchmark} failed at: {run_end}\n")
                    f.write(f"Error: {error_msg}\n")

                # Record failed run
                run_result = _create_multi_benchmark_run_result(
                    config, config_idx, run_name, benchmark, None, run_start, run_end, success=False, error=error_msg
                )
                all_run_results.append(run_result)
                failed_runs += 1

                if not continue_on_failure:
                    logger.error("Stopping sweep due to failure (continue_on_failure=False)")
                    break

        if not continue_on_failure and failed_runs > 0:
            break

    end_time = datetime.now()

    # Create comprehensive summaries
    _create_multi_benchmark_summaries(all_run_results, sweep_dir, benchmarks, configs)

    # Save sweep metadata
    sweep_metadata = {
        "benchmarks": benchmarks,
        "mode": "llm_multi_benchmark_sweep",
        "total_configs": len(configs),
        "total_benchmarks": len(benchmarks),
        "total_runs": len(configs) * len(benchmarks),
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "skipped_runs": skipped_runs,
        "new_runs": len(configs) * len(benchmarks) - skipped_runs,
        "success_rate": successful_runs / (len(configs) * len(benchmarks)) if configs and benchmarks else 0,
        "experiment_name": experiment_name,
        "resumed": resume and skipped_runs > 0,
    }
    save_metadata(sweep_dir, start_time, end_time, sweep_metadata, "sweep_metadata.json")

    # Print final summary
    _print_multi_benchmark_summary(
        sweep_dir, successful_runs, failed_runs, skipped_runs, (end_time - start_time).total_seconds(), benchmarks
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


# =============================================================================
# MULTI-BENCHMARK HELPER FUNCTIONS -------------------------------------------
# =============================================================================


def _generate_multi_benchmark_experiment_name(configs: List[LLMRunConfig], benchmarks: List[str]) -> str:
    """Generate deterministic experiment name for multi-benchmark sweeps."""
    import hashlib

    model_names = sorted(set(c.model_name for c in configs))
    config_summary = {
        "benchmarks": sorted(benchmarks),
        "n_configs": len(configs),
        "learning_rates": sorted(set(c.learning_rate for c in configs)),
        "batch_sizes": sorted(set(c.train_batch_size for c in configs)),
        "lora_ranks": sorted(set(c.lora_rank for c in configs)),
        "model_names": model_names,
    }

    config_str = str(sorted(config_summary.items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    benchmarks_str = "_".join(sorted(benchmarks))
    return f"multi_benchmark_sweep_{benchmarks_str}_model={'+'.join(model_names)}_{config_hash}"


def _validate_multi_benchmark_config(existing_config: Dict[str, Any], new_config: Dict[str, Any]) -> bool:
    """Validate that multi-benchmark configs are compatible for resume."""
    key_fields = ["benchmarks", "n_configs", "configs"]

    for field in key_fields:
        if field not in existing_config or field not in new_config:
            return False

        if field == "configs":
            if len(existing_config[field]) != len(new_config[field]):
                return False

            for old_cfg, new_cfg in zip(existing_config[field], new_config[field]):
                key_params = ["learning_rate", "train_batch_size", "lora_rank", "model_name", "num_epochs"]
                for param in key_params:
                    if old_cfg.get(param) != new_cfg.get(param):
                        return False
        else:
            if field == "benchmarks":
                # Order doesn't matter for benchmarks
                if set(existing_config[field]) != set(new_config[field]):
                    return False
            else:
                if existing_config[field] != new_config[field]:
                    return False

    return True


def _find_completed_multi_benchmark_runs(experiment_dir: Path, benchmarks: List[str]) -> List[Tuple[str, str]]:
    """Find all completed (config, benchmark) pairs in a multi-benchmark experiment."""
    completed_runs = []

    if not experiment_dir.exists():
        return completed_runs

    for benchmark in benchmarks:
        benchmark_dir = experiment_dir / benchmark
        if not benchmark_dir.exists():
            continue

        for item in benchmark_dir.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                # Check if this specific (config, benchmark) run is completed
                required_files = ["results.csv", "metadata.json", "llm_config.json"]

                # Check all required files exist
                all_files_exist = all((item / filename).exists() for filename in required_files)

                if all_files_exist:
                    # Check metadata indicates success
                    try:
                        metadata_path = item / "metadata.json"
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        if metadata.get("success", False):
                            completed_runs.append((item.name, benchmark))
                    except (FileNotFoundError, json.JSONDecodeError, KeyError):
                        continue

    return sorted(completed_runs)


def _run_single_config_benchmark(
    config: LLMRunConfig,
    benchmark: str,
    run_dir: Path,
    n_splits: int,
    random_state: int,
    question_types: Optional[list],
    verbose: bool,
) -> pd.DataFrame:
    """Run evaluation for a single (config, benchmark) combination."""
    # Save config for this run
    save_llm_config(config, run_dir)

    # Load benchmark and data using new system
    benchmark_obj = load_benchmark(benchmark)
    df_full = benchmark_obj.load_data()
    models = benchmark_obj.get_qa_models()  # Use QA model for LLM evaluation
    target_col = get_target_column(benchmark)

    logger.info(f"Loaded {len(df_full)} examples from {benchmark}")
    logger.info(f"Found {len(models)} models: {[m.name for m in models]}")

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


def _create_multi_benchmark_run_result(
    config: LLMRunConfig,
    run_id: int,
    run_name: str,
    benchmark: str,
    summary: Optional[pd.DataFrame],
    start_time: datetime,
    end_time: datetime,
    success: bool,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Create result dictionary for a single (config, benchmark) run."""
    result = {
        "run_id": run_id,
        "run_name": run_name,
        "benchmark": benchmark,
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
        # Extract metrics from summary
        # The summary has columns: Model,Format,Metric,Score,± Std,Count,Feature Importances,Metadata
        score_str = summary.iloc[0]["Score"]  # e.g., "25.0%"
        final_acc = float(score_str.rstrip("%")) / 100  # Convert to decimal
        result["final_acc"] = final_acc
        result["score_display"] = score_str

        # Extract baseline and improvement from metadata if available
        try:
            metadata_str = summary.iloc[0]["Metadata"]
            # Parse the metadata string (it's actually a dict representation)
            import ast

            metadata_dict = ast.literal_eval(metadata_str)
            result["baseline_acc"] = metadata_dict.get("zero_shot_baseline", 0.0)
            result["improvement"] = metadata_dict.get("improvement", 0.0)
        except (ValueError, KeyError, SyntaxError):
            # Fallback if metadata parsing fails
            result["baseline_acc"] = 0.0
            result["improvement"] = final_acc  # Assume improvement is the final score
    else:
        result["final_acc"] = 0.0
        result["baseline_acc"] = 0.0
        result["improvement"] = 0.0
        result["score_display"] = "0.0%" if success else "FAILED"

    if error:
        result["error"] = error

    return result


def _create_multi_benchmark_summaries(
    all_run_results: List[Dict[str, Any]], sweep_dir: Path, benchmarks: List[str], configs: List[LLMRunConfig]
):
    """Create comprehensive summaries for multi-benchmark results."""
    # Convert to DataFrame for easier manipulation
    results_df = pd.DataFrame(all_run_results)

    # 1. Create overall sweep summary (all runs)
    sweep_summary = results_df.sort_values(["benchmark", "final_acc"], ascending=[True, False])
    sweep_summary.to_csv(sweep_dir / "sweep_summary.csv", index=False)
    logger.info(f"Saved complete sweep summary to: {sweep_dir / 'sweep_summary.csv'}")

    # 2. Per-benchmark max performance summary
    benchmark_summary = []
    for benchmark in benchmarks:
        benchmark_runs = results_df[results_df["benchmark"] == benchmark]
        if not benchmark_runs.empty:
            # Find the run with maximum final accuracy for this benchmark
            best_run = benchmark_runs.loc[benchmark_runs["final_acc"].idxmax()]
            benchmark_summary.append(
                {
                    "benchmark": benchmark,
                    "best_run": best_run["run_name"],
                    "best_final_acc": best_run["final_acc"],
                    "best_baseline_acc": best_run["baseline_acc"],
                    "best_improvement": best_run["improvement"],
                    "best_config_lr": best_run["learning_rate"],
                    "best_config_bs": best_run["train_batch_size"],
                    "best_config_rank": best_run["lora_rank"],
                    "best_config_epochs": best_run["num_epochs"],
                }
            )

    benchmark_summary_df = pd.DataFrame(benchmark_summary)
    benchmark_summary_df.to_csv(sweep_dir / "benchmark_summary.csv", index=False)
    logger.info(f"Saved per-benchmark max performance to: {sweep_dir / 'benchmark_summary.csv'}")

    # 3. Per-config cross-benchmark aggregation
    config_summary = []
    for config_idx, config in enumerate(configs, 1):
        run_name = create_run_name(config, config_idx)
        config_runs = results_df[results_df["run_name"] == run_name]

        if not config_runs.empty:
            improvements = config_runs["improvement"].values
            config_summary.append(
                {
                    "run_name": run_name,
                    "config_lr": config.learning_rate,
                    "config_bs": config.train_batch_size,
                    "config_rank": config.lora_rank,
                    "config_epochs": config.num_epochs,
                    "n_benchmarks": len(config_runs),
                    "improvement_mean": improvements.mean(),
                    "improvement_min": improvements.min(),
                    "improvement_max": improvements.max(),
                    "improvement_std": improvements.std(),
                    "benchmarks_tested": ", ".join(sorted(config_runs["benchmark"].values)),
                }
            )

    config_summary_df = pd.DataFrame(config_summary)
    # Sort by mean improvement (descending)
    config_summary_df = config_summary_df.sort_values("improvement_mean", ascending=False)
    config_summary_df.to_csv(sweep_dir / "config_summary.csv", index=False)
    logger.info(f"Saved per-config cross-benchmark stats to: {sweep_dir / 'config_summary.csv'}")


def _print_multi_benchmark_summary(
    sweep_dir: Path,
    successful_runs: int,
    failed_runs: int,
    skipped_runs: int,
    total_duration: float,
    benchmarks: List[str],
):
    """Print formatted multi-benchmark sweep summary."""
    print("\n" + "=" * 80)
    print("MULTI-BENCHMARK SWEEP SUMMARY")
    print("=" * 80)

    # Load and display benchmark summary
    benchmark_summary_path = sweep_dir / "benchmark_summary.csv"
    if benchmark_summary_path.exists():
        benchmark_df = pd.read_csv(benchmark_summary_path)
        print("\n🏆 BEST PERFORMANCE PER BENCHMARK:")
        display_cols = ["benchmark", "best_final_acc", "best_improvement", "best_run"]
        print(benchmark_df[display_cols].to_string(index=False))

    # Load and display top configs
    config_summary_path = sweep_dir / "config_summary.csv"
    if config_summary_path.exists():
        config_df = pd.read_csv(config_summary_path)
        print("\n📊 TOP 5 CONFIGS (BY MEAN IMPROVEMENT):")
        display_cols = [
            "run_name",
            "improvement_mean",
            "improvement_min",
            "improvement_max",
            "config_lr",
            "config_bs",
            "config_rank",
        ]
        print(config_df[display_cols].head(5).to_string(index=False))

        if not config_df.empty:
            best_config = config_df.iloc[0]
            print(f"\n🥇 BEST OVERALL CONFIG (mean improvement): {best_config['run_name']}")
            print(f"   Mean improvement: {best_config['improvement_mean']:.4f}")
            print(
                f"   Min/Max improvement: {best_config['improvement_min']:.4f} / {best_config['improvement_max']:.4f}"
            )
            print(f"   Learning rate: {best_config['config_lr']}")
            print(f"   Batch size: {best_config['config_bs']}")
            print(f"   LoRA rank: {best_config['config_rank']}")

    print("\n📈 OVERALL STATS:")
    print(f"   Benchmarks tested: {len(benchmarks)} ({', '.join(benchmarks)})")
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
