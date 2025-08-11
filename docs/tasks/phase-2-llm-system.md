# Phase 2: Clean LLM System Implementation

## Background

After Phase 1 established base abstractions, we now need to implement a production-ready LLM system for TsT bias detection. The current `llm_utils.py` is a prototype that works but lacks the robustness, scalability, and maintainability needed for production use.

Key requirements for the LLM system:
1. **Efficient k-fold training**: Must handle 5+ LoRA adapters without disk/memory bloat
2. **Parallel inference**: Support multi-GPU inference for speed
3. **Memory management**: Proper GPU memory handling between train/inference phases  
4. **Type safety**: Pydantic models and proper error handling
5. **Extensibility**: Easy to add new LLM models or training strategies

## Objectives

Build a robust, scalable LLM system following DataEnvGym patterns that can handle production workloads with multiple GPUs and large datasets.

## Implementation Plan

### 1. Package Structure

Create the following directory structure:

```
src/TsT/llm/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── models.py          # Pydantic data models
│   └── conversion.py      # Dataset format conversions
├── predictors/
│   ├── __init__.py
│   ├── base.py           # Abstract predictor interfaces
│   ├── vllm.py           # Single-GPU vLLM predictor
│   └── ray_vllm.py       # Multi-GPU Ray-based predictor
├── trainers/
│   ├── __init__.py
│   ├── base.py           # Abstract trainer interfaces
│   └── llamafactory.py   # LlamaFactory wrapper
├── trainable/
│   ├── __init__.py
│   └── predictor.py      # Composed trainable predictor
├── utils/
│   ├── __init__.py
│   ├── io.py             # JSONL I/O utilities
│   └── config.py         # Configuration management
└── configs/
    └── llamafactory/
        └── default_lora.yaml
```

### 2. Data Models

**File**: `src/TsT/llm/data/models.py`

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path

class TrainingDatum(BaseModel):
    """Training data for TsT LLM fine-tuning"""
    instruction: str
    response: str
    metadata: Optional[Dict] = None

class TestInstance(BaseModel):
    """Test instance for TsT LLM inference"""
    instruction: str
    instance_id: str
    ground_truth: str

class LLMPredictionResult(BaseModel):
    """Result from LLM prediction"""
    instance_id: str
    prediction: str
    confidence: Optional[float] = None

class LoRAAdapterInfo(BaseModel):
    """Information about a trained LoRA adapter"""
    fold_id: int
    adapter_path: Path
    training_size: int
    model_name: str
    training_config: Dict
```

### 3. Base Interfaces

**File**: `src/TsT/llm/predictors/base.py`

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from ..data.models import TestInstance, LLMPredictionResult

class LLMPredictorInterface(ABC):
    """Abstract interface for LLM predictors"""
    
    @abstractmethod
    def predict(self, instances: List[TestInstance]) -> List[LLMPredictionResult]:
        """Generate predictions for test instances"""
        pass
    
    @abstractmethod
    def load_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter for fine-tuned inference"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset model state and free GPU memory"""
        pass

class LLMTrainerInterface(ABC):
    """Abstract interface for LLM trainers"""
    
    @abstractmethod
    def train(self, training_data: List[TrainingDatum], output_dir: Path) -> LoRAAdapterInfo:
        """Train LoRA adapter and return adapter info"""
        pass
```

### 4. Single-GPU vLLM Predictor

**File**: `src/TsT/llm/predictors/vllm.py`

```python
from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .base import LLMPredictorInterface
from ..data.models import TestInstance, LLMPredictionResult

@dataclass
class VLLMPredictorConfig:
    model_name: str = "google/gemma-2-2b-it"
    max_seq_length: int = 512
    temperature: float = 0.0
    max_tokens: int = 10
    gpu_memory_utilization: float = 0.8

class VLLMPredictor(LLMPredictorInterface):
    """Single-GPU vLLM predictor with LoRA support"""
    
    def __init__(self, config: VLLMPredictorConfig):
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.llm: Optional[LLM] = None
        self.lora_request: Optional[LoRARequest] = None
        
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=1.0,
        )
        
        self._load_base_model()
    
    def _load_base_model(self):
        """Load base model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.llm = LLM(
            model=self.config.model_name,
            enable_lora=True,
            max_model_len=self.config.max_seq_length,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
    
    def load_adapter(self, adapter_path: str) -> None:
        """Load LoRA adapter for fine-tuned inference"""
        if self.tokenizer is None:
            raise RuntimeError("Base model not loaded")
            
        # Update tokenizer with any new tokens from fine-tuning
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Create LoRA request for vLLM
        self.lora_request = LoRARequest(
            lora_name=f"tst_adapter_{Path(adapter_path).name}",
            lora_int_id=1,
            lora_local_path=adapter_path,
        )
    
    def predict(self, instances: List[TestInstance]) -> List[LLMPredictionResult]:
        """Generate predictions using vLLM"""
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # Format prompts
        prompts = self._format_prompts([inst.instruction for inst in instances])
        
        # Generate responses
        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            use_tqdm=False,
            lora_request=self.lora_request,
        )
        
        # Extract predictions
        results = []
        for instance, output in zip(instances, outputs):
            prediction = output.outputs[0].text.strip()
            # Extract first token for classification tasks
            first_token = prediction.split()[0] if prediction.split() else prediction
            
            results.append(LLMPredictionResult(
                instance_id=instance.instance_id,
                prediction=first_token,
            ))
        
        return results
    
    def _format_prompts(self, instructions: List[str]) -> List[str]:
        """Format instructions using chat template if available"""
        if self.tokenizer.chat_template is not None:
            return [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for instruction in instructions
            ]
        return instructions
    
    def reset(self) -> None:
        """Reset model state and free GPU memory"""
        if self.llm is not None:
            del self.llm
            self.llm = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.lora_request = None
```

### 5. Ray-Based Parallel Predictor

**File**: `src/TsT/llm/predictors/ray_vllm.py`

```python
import ray
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional
from typing_extensions import Self

from .vllm import VLLMPredictor, VLLMPredictorConfig
from .base import LLMPredictorInterface
from ..data.models import TestInstance, LLMPredictionResult
from ..utils.io import PydanticJSONLinesWriter, PydanticJSONLinesReader

@ray.remote(num_gpus=1)
class RayVLLMWorker:
    """Ray worker for parallel vLLM inference"""
    
    def __init__(self, config: VLLMPredictorConfig, worker_index: int):
        self.predictor = VLLMPredictor(config)
        self.worker_index = worker_index
    
    def load_adapter(self, adapter_path: str):
        """Load adapter on this worker"""
        self.predictor.load_adapter(adapter_path)
    
    def predict_batch(self, input_path: str, output_path: str):
        """Process a batch of instances from file"""
        # Read instances
        reader = PydanticJSONLinesReader(input_path, TestInstance)
        instances = list(reader())
        
        # Generate predictions
        results = self.predictor.predict(instances)
        
        # Write results
        writer = PydanticJSONLinesWriter(output_path)
        writer.write_batch(results)

@dataclass 
class RayVLLMPredictorConfig:
    base_config: VLLMPredictorConfig
    num_workers: int = 4
    
    def set_with_gpu_count(self, gpu_count: int) -> Self:
        return replace(self, num_workers=gpu_count)

class RayVLLMPredictor(LLMPredictorInterface):
    """Multi-GPU Ray-based vLLM predictor"""
    
    def __init__(self, config: RayVLLMPredictorConfig):
        self.config = config
        self.workers: Optional[List] = None
    
    def _create_workers(self):
        """Create Ray workers for parallel inference"""
        if self.workers is None:
            self.workers = [
                RayVLLMWorker.remote(self.config.base_config, i)
                for i in range(self.config.num_workers)
            ]
    
    def load_adapter(self, adapter_path: str) -> None:
        """Load adapter on all workers"""
        self._create_workers()
        ray.get([
            worker.load_adapter.remote(adapter_path)
            for worker in self.workers
        ])
    
    def predict(self, instances: List[TestInstance]) -> List[LLMPredictionResult]:
        """Generate predictions using parallel workers"""
        self._create_workers()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            
            # Distribute instances across workers
            worker_chunks = [[] for _ in range(self.config.num_workers)]
            for i, instance in enumerate(instances):
                worker_chunks[i % self.config.num_workers].append(instance)
            
            # Write input files
            input_paths = []
            output_paths = []
            for i, chunk in enumerate(worker_chunks):
                if not chunk:  # Skip empty chunks
                    continue
                    
                input_path = temp_path / f"input_{i}.jsonl"
                output_path = temp_path / f"output_{i}.jsonl"
                
                writer = PydanticJSONLinesWriter(str(input_path))
                writer.write_batch(chunk)
                
                input_paths.append(str(input_path))
                output_paths.append(str(output_path))
            
            # Process in parallel
            ray.get([
                worker.predict_batch.remote(input_path, output_path)
                for worker, input_path, output_path in 
                zip(self.workers[:len(input_paths)], input_paths, output_paths)
            ])
            
            # Collect results
            all_results = []
            for output_path in output_paths:
                reader = PydanticJSONLinesReader(output_path, LLMPredictionResult)
                all_results.extend(reader())
            
            # Maintain original order
            result_by_id = {r.instance_id: r for r in all_results}
            return [result_by_id[inst.instance_id] for inst in instances]
    
    def reset(self) -> None:
        """Reset all workers and free memory"""
        if self.workers is not None:
            for worker in self.workers:
                ray.kill(worker)
            self.workers = None
```

### 6. LlamaFactory Trainer

**File**: `src/TsT/llm/trainers/llamafactory.py`

```python
import tempfile
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from .base import LLMTrainerInterface
from ..data.models import TrainingDatum, LoRAAdapterInfo
from ..utils.llamafactory import (
    format_records_for_llama_factory_sft,
    run_llama_factory_training
)

@dataclass
class LlamaFactoryConfig:
    model_name: str = "google/gemma-2-2b-it"
    template: str = "gemma"
    learning_rate: float = 2e-4
    num_epochs: int = 1
    batch_size: int = 4
    lora_rank: int = 8
    lora_alpha: int = 16
    max_seq_length: int = 512
    fp16: bool = True
    
class LlamaFactoryTrainer(LLMTrainerInterface):
    """LoRA trainer using LlamaFactory"""
    
    def __init__(self, config: LlamaFactoryConfig):
        self.config = config
    
    def train(self, training_data: List[TrainingDatum], output_dir: Path) -> LoRAAdapterInfo:
        """Train LoRA adapter using LlamaFactory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert data to LlamaFactory format
        dataset_dir = output_dir / "dataset"
        data_dicts = [{"instruction": d.instruction, "response": d.response} for d in training_data]
        
        sft_spec, _ = format_records_for_llama_factory_sft(
            data_dicts,
            str(dataset_dir),
            instruction_key="instruction",
            response_key="response",
            overwrite=True,
        )
        
        # Generate training config
        adapter_dir = output_dir / "adapter"
        config_path = self._generate_config(
            dataset_dir=str(dataset_dir),
            dataset_name=sft_spec.dataset_name,
            output_dir=str(adapter_dir),
        )
        
        # Run training
        run_llama_factory_training(config_path)
        
        return LoRAAdapterInfo(
            fold_id=-1,  # Will be set by caller
            adapter_path=adapter_dir,
            training_size=len(training_data),
            model_name=self.config.model_name,
            training_config=self.config.__dict__,
        )
    
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
            
            # Dataset
            "dataset_dir": dataset_dir,
            "dataset": dataset_name,
            "cutoff_len": self.config.max_seq_length,
            "overwrite_cache": True,
            "preprocessing_num_workers": 4,
            
            # Output
            "output_dir": output_dir,
            "logging_steps": 10,
            "save_steps": 500,
            "overwrite_output_dir": True,
            "save_total_limit": 1,
            
            # Training
            "per_device_train_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": 1,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "fp16": self.config.fp16,
            
            # Eval
            "val_size": 0.1,
            "per_device_eval_batch_size": self.config.batch_size,
        }
        
        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f, default_flow_style=False)
            return f.name
```

### 7. Trainable Predictor (Composition)

**File**: `src/TsT/llm/trainable/predictor.py`

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..predictors.base import LLMPredictorInterface
from ..trainers.base import LLMTrainerInterface
from ..data.models import TrainingDatum, TestInstance, LLMPredictionResult

@dataclass
class TrainableLLMPredictorConfig:
    """Configuration for trainable LLM predictor"""
    save_adapters: bool = True
    adapter_prefix: str = "fold"

class TrainableLLMPredictor:
    """Composed trainable predictor following DataEnvGym patterns"""
    
    def __init__(
        self,
        predictor: LLMPredictorInterface,
        trainer: LLMTrainerInterface,
        config: TrainableLLMPredictorConfig = None,
    ):
        self.predictor = predictor
        self.trainer = trainer
        self.config = config or TrainableLLMPredictorConfig()
        self.current_adapter_path: Optional[str] = None
    
    def train(self, training_data: List[TrainingDatum], output_dir: Path) -> None:
        """Train LoRA adapter and load it for inference"""
        # Free GPU memory from inference
        self.predictor.reset()
        
        # Train adapter (uses all GPU memory)
        adapter_info = self.trainer.train(training_data, output_dir)
        
        # Load trained adapter for inference
        self.current_adapter_path = str(adapter_info.adapter_path)
        self.predictor.load_adapter(self.current_adapter_path)
    
    def predict(self, instances: List[TestInstance]) -> List[LLMPredictionResult]:
        """Generate predictions using current adapter"""
        return self.predictor.predict(instances)
    
    def save(self, path: Path) -> None:
        """Save current adapter"""
        if self.current_adapter_path is None:
            raise RuntimeError("No adapter trained yet")
        
        # Copy adapter to save location
        import shutil
        shutil.copytree(self.current_adapter_path, path, dirs_exist_ok=True)
```

### 8. LLM Evaluator Integration

**File**: `src/TsT/core/evaluators.py` (extend from Phase 1)

```python
from typing import Dict, Optional
from ..llm.trainable.predictor import TrainableLLMPredictor
from ..llm.data.conversion import convert_to_tst_format

class LLMEvaluator(ModelEvaluator):
    """Evaluator for LLM-based bias detection"""
    
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config
        self.trainable_predictor: Optional[TrainableLLMPredictor] = None
    
    def evaluate_fold(
        self,
        model: BiasModel,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        fold_num: int,
        seed: int
    ) -> float:
        """Evaluate LLM on a single fold"""
        # Initialize predictor if needed
        if self.trainable_predictor is None:
            self.trainable_predictor = self._create_trainable_predictor()
        
        # Convert data to TsT format
        train_data = convert_to_tst_format(train_df, target_col, model.format)
        test_instances = convert_to_test_instances(test_df, target_col)
        
        # Train on fold
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainable_predictor.train(train_data, Path(temp_dir))
            
            # Evaluate on test set
            predictions = self.trainable_predictor.predict(test_instances)
            
            # Calculate accuracy
            return self._calculate_accuracy(predictions, test_instances, model.format)
```

## Success Criteria

1. **Performance**: Multi-GPU inference provides 3-5x speedup over single GPU
2. **Memory Efficiency**: Proper GPU memory management for k-fold training
3. **Type Safety**: All data flows through Pydantic models
4. **Maintainability**: Clean separation of concerns (predictor/trainer/data)
5. **Extensibility**: Easy to add new predictors or trainers
6. **Production Ready**: Proper error handling, logging, and resource cleanup

## Dependencies to Add

```bash
uv add ray[default]
uv add pydantic
uv add typing-extensions
```

## Testing Strategy

1. **Unit Tests**: Test each component independently
2. **Integration Tests**: Test full pipeline with small datasets  
3. **Performance Tests**: Verify multi-GPU speedup
4. **Memory Tests**: Ensure proper cleanup between folds

## Estimated Effort

**1-2 weeks** - This is a substantial implementation but follows well-established patterns from DataEnvGym.
