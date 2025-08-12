"""
Shared utilities for TsT experiments.

Common functions for experiment management, logging, and result capture.
"""

import json
import sys
import contextlib
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Any

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


def load_benchmark_module(benchmark: str):
    """Dynamically import benchmark module."""
    try:
        import importlib

        return importlib.import_module(f"TsT.benchmarks.{benchmark}")
    except ImportError:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def get_target_column(benchmark: str) -> str:
    """Get target column for benchmark."""
    if benchmark in ["cvb", "video_mme", "mmmu"]:
        return "gt_idx"
    else:
        return "ground_truth"


def create_run_name(config: LLMRunConfig, run_id: int) -> str:
    """Create descriptive name for a single run."""
    return f"run_{run_id:03d}_lr{config.learning_rate}_bs{config.train_batch_size}_rank{config.lora_rank}"
