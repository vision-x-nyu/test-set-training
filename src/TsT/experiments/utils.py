"""
Shared utilities for TsT experiments.

Common functions for experiment management, logging, and result capture.
"""

import json
import sys
import contextlib
from functools import lru_cache
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Any, List

from ezcolorlog import root_logger as logger
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


def create_timestamped_dir(base_dir: str, prefix: str) -> Path:
    """Create timestamped directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_llm_config(config: LLMRunConfig, results_dir: Path, filename: str = "llm_config.json") -> Path:
    """Save LLM configuration to JSON."""
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

    config_path = results_dir / filename
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Saved LLM config to: {config_path}")
    return config_path


def save_metadata(
    results_dir: Path,
    start_time: datetime,
    end_time: datetime,
    metadata: Dict[str, Any],
    filename: str = "metadata.json",
) -> Path:
    """Save experiment metadata."""
    base_metadata = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
    }

    # Merge with provided metadata
    full_metadata = {**base_metadata, **metadata}

    metadata_path = results_dir / filename
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)

    logger.info(f"Saved metadata to: {metadata_path}")
    return metadata_path


def save_logs(
    stdout_capture: StringIO, stderr_capture: StringIO, results_dir: Path, filename: str = "logs.txt"
) -> Path:
    """Save captured logs."""
    logs_path = results_dir / filename
    with open(logs_path, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout_capture.getvalue())
        f.write("\n=== STDERR ===\n")
        f.write(stderr_capture.getvalue())
    logger.info(f"Saved logs to: {logs_path}")
    return logs_path


def save_json(data: Dict[str, Any], results_dir: Path, filename: str) -> Path:
    """Save JSON data to file."""
    file_path = results_dir / filename
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {filename} to: {file_path}")
    return file_path


@lru_cache(maxsize=None)
def load_benchmark(benchmark_name: str):
    """Load a benchmark using the registry system."""
    from ..core.benchmark import BenchmarkRegistry

    return BenchmarkRegistry.get_benchmark(benchmark_name)


def list_available_benchmarks():
    """List all available benchmarks."""
    from ..core.benchmark import BenchmarkRegistry

    return BenchmarkRegistry.list_benchmarks()


def get_target_column(benchmark: str) -> str:
    """Get target column for benchmark."""
    if benchmark in ["cvb", "video_mme", "mmmu"]:
        return "gt_idx"
    else:
        return "ground_truth"


def create_run_name(config: LLMRunConfig, run_id: int) -> str:
    """Create descriptive name for a single run."""
    return f"run_{run_id:03d}_lr{config.learning_rate}_bs{config.train_batch_size}_rank{config.lora_rank}"


def generate_experiment_name(configs: list, benchmark: str) -> str:
    """Generate deterministic experiment name from configs."""
    import hashlib

    # Create deterministic hash from key config parameters
    model_names = sorted(set(c.model_name for c in configs))
    config_summary = {
        "benchmark": benchmark,
        "n_configs": len(configs),
        "learning_rates": sorted(set(c.learning_rate for c in configs)),
        "batch_sizes": sorted(set(c.train_batch_size for c in configs)),
        "lora_ranks": sorted(set(c.lora_rank for c in configs)),
        "model_names": model_names,
    }

    config_str = str(sorted(config_summary.items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    return f"llm_sweep_{benchmark}_model={'+'.join(model_names)}_{config_hash}"


def is_run_completed(run_dir: Path) -> bool:
    """Check if a run completed successfully."""
    required_files = ["results.csv", "metadata.json", "llm_config.json"]

    # Check all required files exist
    for filename in required_files:
        if not (run_dir / filename).exists():
            return False

    # Check metadata indicates success
    try:
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata.get("success", False)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False


def validate_sweep_config(existing_config: Dict[str, Any], new_config: Dict[str, Any]) -> bool:
    """Validate that sweep configs are compatible for resume."""
    # Compare key fields that should match exactly
    key_fields = ["benchmark", "n_configs", "configs"]

    for field in key_fields:
        if field not in existing_config or field not in new_config:
            return False

        # For configs list, compare the essential parameters
        if field == "configs":
            if len(existing_config[field]) != len(new_config[field]):
                return False

            for old_cfg, new_cfg in zip(existing_config[field], new_config[field]):
                # Compare key hyperparameters
                key_params = ["learning_rate", "train_batch_size", "lora_rank", "model_name", "num_epochs"]
                for param in key_params:
                    if old_cfg.get(param) != new_cfg.get(param):
                        return False
        else:
            if existing_config[field] != new_config[field]:
                return False

    return True


def find_completed_runs(experiment_dir: Path) -> List[str]:
    """Find all completed runs in an experiment directory."""
    completed_runs = []

    if not experiment_dir.exists():
        return completed_runs

    for item in experiment_dir.iterdir():
        if item.is_dir() and item.name.startswith("run_"):
            if is_run_completed(item):
                completed_runs.append(item.name)

    return sorted(completed_runs)
