"""
Ray-based parallel vLLM predictor for multi-GPU inference.

This module implements a distributed vLLM predictor using Ray for parallel inference
across multiple GPUs, following DataEnvGym patterns for scalability.
"""

import ray
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional
from typing_extensions import Self

from .vllm import VLLMPredictor, VLLMPredictorConfig
from .base import BaseLLMPredictor
from ..data.models import TestInstance, LLMPredictionResult
from ..utils.io import PydanticJSONLinesWriter, PydanticJSONLinesReader


@ray.remote(num_gpus=1)
class RayVLLMWorker:
    """Ray worker for parallel vLLM inference"""

    def __init__(self, config: VLLMPredictorConfig, worker_index: int):
        """
        Initialize Ray worker with vLLM predictor.

        Args:
            config: Configuration for the vLLM predictor
            worker_index: Index of this worker (for logging/debugging)
        """
        self.predictor = VLLMPredictor(config)
        self.worker_index = worker_index

    def load_adapter(self, adapter_path: str) -> bool:
        """
        Load adapter on this worker.

        Args:
            adapter_path: Path to the LoRA adapter

        Returns:
            True if successful, False otherwise
        """
        try:
            self.predictor.load_adapter(adapter_path)
            return True
        except Exception as e:
            ray.get_runtime_context().get_assigned_resources()
            print(f"Worker {self.worker_index} failed to load adapter: {e}")
            return False

    def predict_batch(self, input_path: str, output_path: str) -> bool:
        """
        Process a batch of instances from file.

        Args:
            input_path: Path to JSONL file with TestInstance objects
            output_path: Path to write LLMPredictionResult objects

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read instances from file
            reader = PydanticJSONLinesReader(input_path, TestInstance)
            instances = list(reader())

            if not instances:
                # Write empty results file
                writer = PydanticJSONLinesWriter(output_path)
                writer.write_batch([])
                return True

            # Generate predictions
            results = self.predictor.predict(instances)

            # Write results to file
            writer = PydanticJSONLinesWriter(output_path)
            writer.write_batch(results)

            return True

        except Exception as e:
            print(f"Worker {self.worker_index} failed to process batch: {e}")
            return False

    def get_status(self) -> dict:
        """Get worker status information"""
        return {
            "worker_index": self.worker_index,
            "is_loaded": self.predictor.is_loaded,
            "current_adapter": self.predictor.current_adapter_path,
        }

    def reset(self) -> None:
        """Reset the predictor and free memory"""
        self.predictor.reset()


@dataclass
class RayVLLMPredictorConfig:
    """Configuration for Ray-based parallel vLLM predictor"""

    base_config: VLLMPredictorConfig
    num_workers: int = 4

    def set_with_gpu_count(self, gpu_count: int) -> Self:
        """Create a new config with the specified number of workers"""
        return replace(self, num_workers=gpu_count)


class RayVLLMPredictor(BaseLLMPredictor):
    """Multi-GPU Ray-based vLLM predictor"""

    def __init__(self, config: RayVLLMPredictorConfig):
        super().__init__()
        self.config = config
        self.workers: Optional[List] = None

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def _create_workers(self) -> None:
        """Create Ray workers for parallel inference"""
        if self.workers is None:
            try:
                self.workers = [
                    RayVLLMWorker.remote(self.config.base_config, i) for i in range(self.config.num_workers)
                ]
                self._set_loaded(True)
            except Exception as e:
                self._set_loaded(False)
                raise RuntimeError(f"Failed to create Ray workers: {e}")

    def load_adapter(self, adapter_path: str) -> None:
        """Load adapter on all workers"""
        self._create_workers()

        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        # Load adapter on all workers in parallel
        load_tasks = [worker.load_adapter.remote(adapter_path) for worker in self.workers]

        # Wait for all workers to load the adapter
        results = ray.get(load_tasks)

        # Check if all workers loaded successfully
        failed_workers = [i for i, success in enumerate(results) if not success]
        if failed_workers:
            raise RuntimeError(f"Failed to load adapter on workers: {failed_workers}")

        self._set_adapter_path(adapter_path)

    def predict(self, instances: List[TestInstance]) -> List[LLMPredictionResult]:
        """Generate predictions using parallel workers"""
        if not instances:
            return []

        self._validate_instances(instances)
        self._create_workers()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)

            # Distribute instances across workers
            worker_chunks = [[] for _ in range(self.config.num_workers)]
            for i, instance in enumerate(instances):
                worker_idx = i % self.config.num_workers
                worker_chunks[worker_idx].append(instance)

            # Prepare input/output files for each worker
            worker_tasks = []
            active_workers = []

            for i, (worker, chunk) in enumerate(zip(self.workers, worker_chunks)):
                if not chunk:  # Skip workers with no data
                    continue

                input_path = temp_path / f"input_{i}.jsonl"
                output_path = temp_path / f"output_{i}.jsonl"

                # Write input file
                writer = PydanticJSONLinesWriter(str(input_path))
                writer.write_batch(chunk)

                # Submit task to worker
                task = worker.predict_batch.remote(str(input_path), str(output_path))
                worker_tasks.append((task, str(output_path)))
                active_workers.append(i)

            # Wait for all workers to complete
            task_results = ray.get([task for task, _ in worker_tasks])

            # Check for failures
            failed_tasks = [i for i, success in enumerate(task_results) if not success]
            if failed_tasks:
                failed_worker_indices = [active_workers[i] for i in failed_tasks]
                raise RuntimeError(f"Prediction failed on workers: {failed_worker_indices}")

            # Collect results from all workers
            all_results = []
            for _, output_path in worker_tasks:
                if Path(output_path).exists():
                    reader = PydanticJSONLinesReader(output_path, LLMPredictionResult)
                    worker_results = list(reader())
                    all_results.extend(worker_results)

            # Restore original order based on instance IDs
            result_by_id = {r.instance_id: r for r in all_results}
            ordered_results = [result_by_id[inst.instance_id] for inst in instances]

            return ordered_results

    def reset(self) -> None:
        """Reset all workers and free memory"""
        if self.workers is not None:
            try:
                # Reset all workers
                reset_tasks = [worker.reset.remote() for worker in self.workers]
                ray.get(reset_tasks)

                # Kill all workers
                for worker in self.workers:
                    ray.kill(worker)

            except Exception as e:
                print(f"Warning: Error during worker cleanup: {e}")
            finally:
                self.workers = None

        self._set_adapter_path(None)
        self._set_loaded(False)

    def get_worker_status(self) -> List[dict]:
        """Get status of all workers"""
        if self.workers is None:
            return []

        status_tasks = [worker.get_status.remote() for worker in self.workers]
        return ray.get(status_tasks)

    @property
    def num_active_workers(self) -> int:
        """Get number of active workers"""
        return len(self.workers) if self.workers else 0

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.reset()
        except Exception:
            # Ignore cleanup errors during destruction
            pass
