"""
LlamaFactory trainer for LoRA fine-tuning.

This module implements a LoRA trainer using LlamaFactory, following DataEnvGym patterns.
It includes small helpers for preparing LlamaFactory datasets and running the CLI.
"""

import tempfile
import yaml
import os
import json
import subprocess
import sh
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaseLLMTrainer, ProgressCallback
from ..data.models import TrainingDatum, LoRAAdapterInfo
from pydantic import BaseModel
from ezcolorlog import root_logger as logger


llamafactory_cli = sh.Command("llamafactory-cli")
lf_trainer = llamafactory_cli.bake("train")


class AlpacaFormatExpectedKeysPerRecord(BaseModel):
    prompt: str = "instruction"
    response: str = "output"

    def adapt_record_to_spec(self, record: dict, prompt_key: str, response_key: str) -> dict:
        """Convert a record to match the expected format of the LlamaFactory finetuning spec."""
        try:
            new_record = {
                self.prompt: record[prompt_key],
                self.response: record[response_key],
            }
        except KeyError as e:
            raise KeyError(
                f"Record has keys {record.keys()} but expected keys "
                f"{self.prompt} -> {prompt_key}, {self.response} -> {response_key} "
                f"could not access {e}"
            )
        return new_record


DEFAULT_DATASET_FILE_NAME = "custom_sft_dataset.jsonl"


class LlamaFactoryMinimalSftSpec(BaseModel):
    file_name: str = DEFAULT_DATASET_FILE_NAME
    formatting: str = "alpaca"
    ranking: bool = False
    load_from: str = "file"
    columns: AlpacaFormatExpectedKeysPerRecord = AlpacaFormatExpectedKeysPerRecord()

    @property
    def dataset_name(self) -> str:
        """The dataset name LlamaFactory will refer to this dataset as."""
        return os.path.splitext(self.file_name)[0]

    @property
    def llama_factory_dataset_info_entry(self) -> dict:
        """Generate the dataset_info.json entry for the LlamaFactory dataset."""
        dataset_name = self.dataset_name
        return {dataset_name: self.model_dump(exclude={"dataset_name"})}

    def get_path_to_dataset_records(self, llama_factory_dataset_dir: str) -> str:
        """Get the path to the dataset records."""
        return os.path.join(llama_factory_dataset_dir, self.file_name)


def format_records_for_llama_factory_sft(
    jsonl_records_or_collection: list[dict],
    llama_factory_dataset_dir: str,
    instruction_key: str,
    response_key: str,
    overwrite: bool = True,
) -> tuple[LlamaFactoryMinimalSftSpec, str]:
    """
    Prepare a collection of records for LlamaFactory supervised fine-tuning.
    """
    supervised_finetuning_spec = LlamaFactoryMinimalSftSpec()

    # Make LlamaFactory dataset directory if it doesn't exist
    os.makedirs(llama_factory_dataset_dir, exist_ok=True)

    with open(os.path.join(llama_factory_dataset_dir, "dataset_info.json"), "w") as f:
        json.dump(supervised_finetuning_spec.llama_factory_dataset_info_entry, f)

    records = jsonl_records_or_collection
    adapted_records = [
        supervised_finetuning_spec.columns.adapt_record_to_spec(record, instruction_key, response_key)
        for record in records
    ]

    records_output_path = supervised_finetuning_spec.get_path_to_dataset_records(llama_factory_dataset_dir)

    mode = "w" if overwrite else "a"
    with open(records_output_path, mode) as f:
        for record in adapted_records:
            json.dump(record, f)
            f.write("\n")

    return supervised_finetuning_spec, records_output_path


def run_llama_factory_training(config_path: str, cuda_visible_devices: Optional[list[int]] = None):
    """
    Run LlamaFactory training with the given configuration path.
    """
    # Set up environment
    _env = os.environ.copy()
    if cuda_visible_devices is not None:
        _env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))

    # Run training command
    try:
        result = lf_trainer(config_path, _out=sys.stdout, _err=sys.stderr, _env=_env)
        logger.info(f"Training completed successfully for {config_path}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    finally:
        # Clean up config file
        if os.path.exists(config_path):
            os.unlink(config_path)


@dataclass
class LlamaFactoryConfig:
    """Configuration for LlamaFactory trainer"""

    model_name: str = "google/gemma-2-2b-it"
    template: str = "gemma"
    learning_rate: float = 2e-4
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    max_seq_length: int = 512
    fp16: bool = True
    seed: int = 42

    # Evaluation settings
    val_size: float = 0.1
    evaluation_strategy: str = "steps"
    eval_steps: int = 100

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 1
    overwrite_output_dir: bool = True

    # Memory and performance
    dataloader_num_workers: int = 4
    preprocessing_num_workers: int = 4

    # GPU settings
    cuda_visible_devices: Optional[List[int]] = None


class LlamaFactoryTrainer(BaseLLMTrainer):
    """LoRA trainer using LlamaFactory"""

    def __init__(self, config: LlamaFactoryConfig, progress_callback: Optional[ProgressCallback] = None):
        super().__init__(progress_callback)
        self.config = config

    def train(self, training_data: List[TrainingDatum], output_dir: Path) -> LoRAAdapterInfo:
        """Train LoRA adapter using LlamaFactory"""
        # Validate training data
        self.validate_training_data(training_data)

        # Prepare output directory
        output_dir = self._prepare_output_directory(output_dir)

        try:
            self._set_training_state(True)

            # Convert training data to LlamaFactory format
            dataset_dir = output_dir / "dataset"
            adapter_dir = output_dir / "adapter"

            data_dicts = [{"instruction": datum.instruction, "response": datum.response} for datum in training_data]

            sft_spec, _ = format_records_for_llama_factory_sft(
                data_dicts,
                str(dataset_dir),
                instruction_key="instruction",
                response_key="response",
                overwrite=True,
            )

            # Generate training config
            config_path = self._generate_config(
                dataset_dir=str(dataset_dir),
                dataset_name=sft_spec.dataset_name,
                output_dir=str(adapter_dir),
            )

            # Notify training start
            estimated_steps = self._estimate_training_steps(len(training_data))
            self._notify_training_start(estimated_steps)

            # Run training
            run_llama_factory_training(config_path, cuda_visible_devices=self.config.cuda_visible_devices)

            # Notify completion
            self._notify_training_end({"final_loss": 0.0})  # LlamaFactory doesn't return metrics

            return LoRAAdapterInfo(
                fold_id=-1,  # Will be set by caller
                adapter_path=adapter_dir,
                training_size=len(training_data),
                model_name=self.config.model_name,
                training_config=self._get_training_config_dict(),
            )

        finally:
            self._set_training_state(False)

    def _generate_config(self, dataset_dir: str, dataset_name: str, output_dir: str) -> str:
        """Generate LlamaFactory YAML config"""
        config = {
            # Model
            "model_name_or_path": self.config.model_name,
            "template": self.config.template,
            # Method
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_target": "all",
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            # Dataset
            "dataset_dir": dataset_dir,
            "dataset": dataset_name,
            "cutoff_len": self.config.max_seq_length,
            "overwrite_cache": True,
            "preprocessing_num_workers": self.config.preprocessing_num_workers,
            # Output
            "output_dir": output_dir,
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "overwrite_output_dir": self.config.overwrite_output_dir,
            "save_total_limit": self.config.save_total_limit,
            # Training
            "per_device_train_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "fp16": self.config.fp16,
            "seed": self.config.seed,
            # Evaluation
            "val_size": self.config.val_size,
            "evaluation_strategy": self.config.evaluation_strategy,
            "eval_steps": self.config.eval_steps,
            "per_device_eval_batch_size": self.config.batch_size,
            # Performance
            "dataloader_num_workers": self.config.dataloader_num_workers,
        }

        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f, default_flow_style=False)
            return f.name

    def _estimate_training_steps(self, num_examples: int) -> int:
        """Estimate total number of training steps"""
        # Calculate steps per epoch
        effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps

        # Account for train/val split
        train_examples = int(num_examples * (1.0 - self.config.val_size))
        steps_per_epoch = max(1, train_examples // effective_batch_size)

        total_steps = steps_per_epoch * self.config.num_epochs
        return total_steps

    def _get_training_config_dict(self) -> Dict[str, Any]:
        """Get training configuration as dictionary"""
        return {
            "model_name": self.config.model_name,
            "learning_rate": self.config.learning_rate,
            "num_epochs": self.config.num_epochs,
            "batch_size": self.config.batch_size,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "max_seq_length": self.config.max_seq_length,
            "template": self.config.template,
        }


class TrainingProgressMonitor(ProgressCallback):
    """Simple progress monitor for LlamaFactory training"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = None

    def on_training_start(self, total_steps: int) -> None:
        """Called when training starts"""
        import time

        self.start_time = time.time()
        if self.verbose:
            print(f"Starting training for {total_steps} steps...")

    def on_training_end(self, final_metrics: Dict[str, float]) -> None:
        """Called when training completes"""
        if self.verbose and self.start_time:
            import time

            duration = time.time() - self.start_time
            print(f"Training completed in {duration:.2f} seconds")
            if final_metrics:
                print(f"Final metrics: {final_metrics}")


def create_llamafactory_trainer(
    model_name: str = "google/gemma-2-2b-it",
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    batch_size: int = 4,
    lora_rank: int = 8,
    max_seq_length: int = 512,
    verbose: bool = False,
    **kwargs,
) -> LlamaFactoryTrainer:
    """
    Convenience function to create a LlamaFactory trainer with common settings.

    Args:
        model_name: Base model to fine-tune
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        batch_size: Training batch size
        lora_rank: LoRA rank parameter
        max_seq_length: Maximum sequence length
        verbose: Whether to print training progress
        **kwargs: Additional configuration parameters

    Returns:
        Configured LlamaFactoryTrainer instance
    """
    config = LlamaFactoryConfig(
        model_name=model_name,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lora_rank=lora_rank,
        max_seq_length=max_seq_length,
        **kwargs,
    )

    progress_callback = TrainingProgressMonitor(verbose=verbose) if verbose else None

    return LlamaFactoryTrainer(config, progress_callback)
