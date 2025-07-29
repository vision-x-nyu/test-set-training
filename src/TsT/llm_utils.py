"""
LLM utilities for TsT evaluation using LLaMA-Factory integration.

This module provides utilities for:
1. Converting datasets to LLaMA-Factory format
2. Generating training configurations
3. Running LoRA fine-tuning
4. Loading and using trained models for inference

Based on the implementation of DataEnvGym: https://github.com/codezakh/DataEnvGym
"""

# https://github.com/vllm-project/vllm/issues/7151
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import os
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer
from ezcolorlog import root_logger as logger
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


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


def generate_llama_factory_config(
    dataset_dir: str,
    dataset_name: str,
    output_dir: str,
    model_name: str = "google/gemma-2-2b-it",
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    batch_size: int = 4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    max_seq_length: int = 512,
    seed: int = 42,
) -> str:
    """
    Generate a LLaMA-Factory configuration file for LoRA fine-tuning.
    """
    config = {
        # Model
        "model_name_or_path": model_name,
        # Method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        # Dataset
        "dataset_dir": dataset_dir,
        "dataset": dataset_name,
        "cutoff_len": max_seq_length,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        # Output
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 500,
        "overwrite_output_dir": True,
        "save_total_limit": 1,
        # Training
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "fp16": True,
        # Eval
        "val_size": 0.1,
        "per_device_eval_batch_size": batch_size,
        # Other
        "seed": seed,
    }

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name

    return config_path


def run_llama_factory_training(config_path: str, cuda_visible_devices: Optional[List[int]] = None):
    """
    Run LLaMA-Factory training with the given configuration.
    """
    # Set up environment
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))

    # Run training command
    # Use the llamafactory-cli command from our LLaMA-Factory installation
    llamafactory_path = Path(__file__).parent.parent / "external" / "LLaMA-Factory" / "src" / "train.py"

    cmd = ["python", str(llamafactory_path), config_path]

    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
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


class LLMPredictor:
    """
    LLM predictor that can load base models and LoRA adapters for inference.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        batch_size: int = 4,
        max_seq_length: int = 512,
        temperature: float = 0.0,  # Use deterministic generation
        max_tokens: int = 10,  # Short answers for TsT
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Model components
        self.tokenizer = None
        self.model = None
        self.llm = None
        self.lora_request = None

        # Sampling params for VLLM
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )

        self._load_base_model()

    def _load_base_model(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading base model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Use VLLM for efficient inference
        self.llm = LLM(
            model=self.model_name,
            enable_lora=True,
            max_model_len=self.max_seq_length,
            gpu_memory_utilization=0.8,
        )

    def load_adapter(self, adapter_path: str):
        """Load a LoRA adapter for fine-tuned inference."""
        logger.info(f"Loading LoRA adapter: {adapter_path}")

        # Update tokenizer to include any new tokens from fine-tuning
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        # Create LoRA request for VLLM
        self.lora_request = LoRARequest(
            lora_name=f"tst_adapter_{os.path.basename(adapter_path)}",
            lora_int_id=1,
            lora_local_path=adapter_path,
        )

    def predict(self, instructions: List[str]) -> List[str]:
        """
        Generate predictions for a list of instructions.
        """
        if not instructions:
            return []

        # Format prompts using chat template if available
        if self.tokenizer.chat_template is not None:
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for instruction in instructions
            ]
        else:
            prompts = instructions

        # Generate in batches
        all_predictions = []

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # Generate responses
            outputs = self.llm.generate(
                batch_prompts,
                self.sampling_params,
                use_tqdm=False,
                lora_request=self.lora_request,
            )

            # Extract generated text
            batch_predictions = []
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                # Take only the first token/word for classification tasks
                first_token = generated_text.split()[0] if generated_text.split() else generated_text
                batch_predictions.append(first_token)

            all_predictions.extend(batch_predictions)

        return all_predictions

    def reset(self):
        """Reset the model state and free GPU memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.lora_request = None
