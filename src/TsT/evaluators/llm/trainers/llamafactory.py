"""
LoRA fine-tuning with LLaMA-Factory (https://github.com/hiyouga/LlamaFactory).

For each fold the trainer writes the training examples as an alpaca-format dataset with
its ``dataset_info.json``, writes a training config (the App. D.1 settings; see
``configs/app_d1/llamafactory_train.yaml``), and runs ``python -m llamafactory.cli train``
in a subprocess on one GPU.

The dataset preparation and the subprocess call are adapted from DataEnvGym
(https://github.com/codezakh/dataenvgym, commit f698f39, ``llama_factory_utils.py``),
MIT License, Copyright (c) 2024 Zaid Khan; see THIRD_PARTY_NOTICES.md.
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from ..data.models import LoRAAdapterInfo, TrainingDatum

logger = logging.getLogger(__name__)

DATASET_NAME = "custom_sft_dataset"
CONFIG_FILE_NAME = "llamafactory_train.yaml"


@dataclass
class LlamaFactoryConfig:
    """LoRA training settings (defaults: App. D.1)."""

    model_name: str = "Qwen/Qwen2-7B-Instruct"
    model_revision: Optional[str] = None
    template: str = "gemma"
    learning_rate: float = 2e-4
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    lora_rank: int = 8
    lora_alpha: Optional[int] = 16  # None: 2 * lora_rank
    lora_dropout: float = 0.1
    max_seq_length: int = 1024
    fp16: bool = True
    seed: int = 42
    val_size: float = 0.1
    eval_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 1
    overwrite_output_dir: bool = True
    dataloader_num_workers: int = 4
    preprocessing_num_workers: int = 4


def write_alpaca_dataset(training_data: List[TrainingDatum], dataset_dir: Path) -> str:
    """Write the examples and their ``dataset_info.json`` entry; return the dataset name."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{DATASET_NAME}.jsonl"
    info = {
        DATASET_NAME: {
            "file_name": file_name,
            "formatting": "alpaca",
            "ranking": False,
            "load_from": "file",
            "columns": {"prompt": "instruction", "response": "output"},
        }
    }
    (dataset_dir / "dataset_info.json").write_text(json.dumps(info))
    with open(dataset_dir / file_name, "w") as f:
        for datum in training_data:
            json.dump({"instruction": datum.instruction, "output": datum.response}, f)
            f.write("\n")
    return DATASET_NAME


def training_config(config: LlamaFactoryConfig, dataset_dir: str, dataset_name: str, output_dir: str) -> dict:
    """The LLaMA-Factory training config for one fold."""
    cfg = {
        "model_name_or_path": config.model_name,
        "template": config.template,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha if config.lora_alpha is not None else 2 * config.lora_rank,
        "lora_dropout": config.lora_dropout,
        "dataset_dir": dataset_dir,
        "dataset": dataset_name,
        "cutoff_len": config.max_seq_length,
        "overwrite_cache": True,
        "preprocessing_num_workers": config.preprocessing_num_workers,
        "output_dir": output_dir,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "overwrite_output_dir": config.overwrite_output_dir,
        "save_total_limit": config.save_total_limit,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "fp16": config.fp16,
        "seed": config.seed,
        "val_size": config.val_size,
        "eval_steps": config.eval_steps,
        "per_device_eval_batch_size": config.batch_size,
        "dataloader_num_workers": config.dataloader_num_workers,
    }
    if config.model_revision is not None:
        cfg["model_revision"] = config.model_revision
    return cfg


def single_gpu_env() -> dict:
    """Subprocess environment with exactly one visible GPU.

    With several visible GPUs, LLaMA-Factory launches distributed training on all of
    them, which multiplies the effective batch size. The paper's runs used one GPU.
    """
    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES")
    devices = [d.strip() for d in (visible or "").split(",") if d.strip()]
    if visible is None or len(devices) > 1:
        env["CUDA_VISIBLE_DEVICES"] = devices[0] if devices else "0"
        logger.info(f"LoRA training runs on one GPU (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
    return env


def run_llamafactory_training(config_path: Path) -> None:
    """Run ``python -m llamafactory.cli train <config>`` with this interpreter; raise on failure."""
    cmd = [sys.executable, "-m", "llamafactory.cli", "train", str(config_path)]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=single_gpu_env())
    if result.returncode != 0:
        raise RuntimeError(f"LLaMA-Factory training failed (exit status {result.returncode}): {' '.join(cmd)}")


class LlamaFactoryTrainer:
    """Trains one LoRA adapter per call."""

    def __init__(self, config: LlamaFactoryConfig):
        self.config = config

    def train(self, training_data: List[TrainingDatum], output_dir: Path) -> LoRAAdapterInfo:
        """Train on ``training_data``; the adapter is written to ``output_dir/adapter``."""
        import yaml

        if not training_data:
            raise ValueError("No training data provided")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "dataset"
        adapter_dir = output_dir / "adapter"
        dataset_name = write_alpaca_dataset(training_data, dataset_dir)
        cfg = training_config(self.config, str(dataset_dir), dataset_name, str(adapter_dir))
        config_path = output_dir / CONFIG_FILE_NAME
        config_path.write_text(yaml.dump(cfg, default_flow_style=False))
        run_llamafactory_training(config_path)
        if not adapter_dir.exists():
            raise RuntimeError(f"LLaMA-Factory finished but wrote no adapter to {adapter_dir}")
        return LoRAAdapterInfo(
            adapter_path=adapter_dir,
            training_size=len(training_data),
            model_name=self.config.model_name,
            training_config=asdict(self.config),
        )
