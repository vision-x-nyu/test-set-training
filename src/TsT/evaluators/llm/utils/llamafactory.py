"""
LlamaFactory integration utilities for TsT LLM training.

This module provides utilities for:
1. Converting datasets to LlamaFactory format
2. Generating training configurations
3. Running LoRA fine-tuning via LlamaFactory

Based on the implementation of DataEnvGym: https://github.com/codezakh/DataEnvGym
"""

# https://github.com/vllm-project/vllm/issues/7151
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import os
import subprocess
import sh
import sys
from typing import Dict, List, Optional
from pydantic import BaseModel

from ezcolorlog import root_logger as logger


llamafactory_cli = sh.Command("llamafactory-cli")
lf_trainer = llamafactory_cli.bake("train")


class AlpacaFormatExpectedKeysPerRecord(BaseModel):
    prompt: str = "instruction"
    response: str = "output"

    def adapt_record_to_spec(self, record: dict, prompt_key: str, response_key: str) -> dict:
        """Convert a record to match the expected format of the llama-factory finetuning spec."""
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
        """The dataset name llama-factory will refer to this dataset as."""
        return os.path.splitext(self.file_name)[0]

    @property
    def llama_factory_dataset_info_entry(self) -> dict:
        """Generate the dataset_info.json entry for the llama-factory dataset."""
        dataset_name = self.dataset_name
        return {dataset_name: self.model_dump(exclude={"dataset_name"})}

    def get_path_to_dataset_records(self, llama_factory_dataset_dir: str) -> str:
        """Get the path to the dataset records."""
        return os.path.join(llama_factory_dataset_dir, self.file_name)


def format_records_for_llama_factory_sft(
    jsonl_records_or_collection: List[Dict[str, str]],
    llama_factory_dataset_dir: str,
    instruction_key: str,
    response_key: str,
    overwrite: bool = True,
) -> tuple[LlamaFactoryMinimalSftSpec, str]:
    """
    Take a collection of records and prepare it for llama-factory finetuning.
    """
    supervised_finetuning_spec = LlamaFactoryMinimalSftSpec()

    # Make llama-factory dataset directory if it doesn't exist
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


def run_llama_factory_training(config_path: str, cuda_visible_devices: Optional[List[int]] = None):
    """
    Run LLaMA-Factory training with the given configuration.
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
