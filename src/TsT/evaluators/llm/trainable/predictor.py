"""
Composed trainable predictor following DataEnvGym patterns.

This module implements a trainable predictor that composes a predictor and trainer,
following the composition-over-inheritance pattern used in DataEnvGym for clean
separation of concerns and GPU memory management.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..predictors.base import LLMPredictorInterface
from ..trainers.base import LLMTrainerInterface
from ..data.models import TrainingDatum, TestInstance, LLMPredictionResult, LoRAAdapterInfo


@dataclass
class TrainableLLMPredictorConfig:
    """Configuration for trainable LLM predictor"""

    save_adapters: bool = True
    adapter_prefix: str = "fold"
    cleanup_after_training: bool = True

    # Memory management settings
    reset_predictor_before_training: bool = True
    reset_predictor_after_training: bool = False


class TrainableLLMPredictor:
    """
    Composed trainable predictor following DataEnvGym patterns.

    This class composes a predictor and trainer to provide both training and inference
    capabilities while managing GPU memory efficiently for k-fold cross-validation.

    Key design principles:
    1. Composition over inheritance
    2. Clear separation of training and inference concerns
    3. Proper GPU memory management between training and inference
    4. Stateless design for k-fold compatibility
    """

    def __init__(
        self,
        predictor: LLMPredictorInterface,
        trainer: LLMTrainerInterface,
        config: Optional[TrainableLLMPredictorConfig] = None,
    ):
        """
        Initialize trainable predictor.

        Args:
            predictor: LLM predictor for inference
            trainer: LLM trainer for fine-tuning
            config: Configuration for trainable predictor behavior
        """
        self.predictor = predictor
        self.trainer = trainer
        self.config = config or TrainableLLMPredictorConfig()

        # Track current state
        self.current_adapter_path: Optional[str] = None
        self._last_training_info: Optional[LoRAAdapterInfo] = None

    def train(self, training_data: List[TrainingDatum], output_dir: Path) -> LoRAAdapterInfo:
        """
        Train LoRA adapter and optionally load it for inference.

        Args:
            training_data: List of training examples
            output_dir: Directory to save the trained adapter

        Returns:
            LoRAAdapterInfo with details about the trained adapter
        """
        if not training_data:
            raise ValueError("No training data provided")

        # Prepare output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Free GPU memory from inference before training
            if self.config.reset_predictor_before_training:
                self.predictor.reset()

            # Run training (this will use all available GPU memory)
            adapter_info = self.trainer.train(training_data, output_dir)

            # Store training info
            self._last_training_info = adapter_info

            # Ensure predictor is loaded (it may have been reset before training)
            try:
                self.predictor.ensure_loaded()
            except Exception as e:
                if self.config.reset_predictor_after_training:
                    self.predictor.reset()
                raise RuntimeError(f"Failed to ensure predictor is loaded before loading adapter: {e}")

            # Load trained adapter for inference
            self._load_adapter_for_inference(str(adapter_info.adapter_path))

            return adapter_info

        except Exception as e:
            # Ensure predictor is reset on training failure
            if self.config.reset_predictor_after_training:
                self.predictor.reset()
            raise RuntimeError(f"Training failed: {e}")

    def predict(self, instances: List[TestInstance]) -> List[LLMPredictionResult]:
        """
        Generate predictions using current adapter.

        Args:
            instances: List of test instances to predict on

        Returns:
            List of prediction results
        """
        if not self.predictor.is_loaded:
            raise RuntimeError("Predictor not loaded. Must train or load adapter first.")

        return self.predictor.predict(instances)

    def load_adapter(self, adapter_path: str) -> None:
        """
        Load a pre-trained adapter for inference.

        Args:
            adapter_path: Path to the LoRA adapter directory
        """
        self._load_adapter_for_inference(adapter_path)

    def save_current_adapter(self, save_path: Path) -> None:
        """
        Save the current adapter to a new location.

        Args:
            save_path: Path where the adapter should be saved

        Raises:
            RuntimeError: If no adapter is currently loaded
        """
        if self.current_adapter_path is None:
            raise RuntimeError("No adapter currently loaded")

        # Copy adapter to save location
        import shutil

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            shutil.rmtree(save_path)

        shutil.copytree(self.current_adapter_path, save_path)

    def reset(self) -> None:
        """
        Reset both predictor and trainer, freeing all GPU memory.

        This is important for k-fold cross-validation where we need to
        completely clean up between folds.
        """
        # Reset predictor
        self.predictor.reset()

        # Stop trainer if running
        if self.trainer.is_training:
            self.trainer.stop_training()

        # Clear state
        self.current_adapter_path = None
        self._last_training_info = None

    def _load_adapter_for_inference(self, adapter_path: str) -> None:
        """
        Load adapter for inference and update tracking.

        Args:
            adapter_path: Path to the adapter to load
        """
        self.predictor.load_adapter(adapter_path)
        self.current_adapter_path = adapter_path

    @property
    def is_training(self) -> bool:
        """Check if training is currently in progress"""
        return self.trainer.is_training

    @property
    def is_loaded(self) -> bool:
        """Check if predictor is loaded and ready for inference"""
        return self.predictor.is_loaded

    @property
    def last_training_info(self) -> Optional[LoRAAdapterInfo]:
        """Get information about the last training run"""
        return self._last_training_info

    def get_memory_usage_info(self) -> dict:
        """
        Get information about current memory usage.

        Returns:
            Dictionary with memory usage information
        """
        import torch

        info = {
            "predictor_loaded": self.is_loaded,
            "trainer_active": self.is_training,
            "current_adapter": self.current_adapter_path,
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "gpu_memory_allocated": torch.cuda.memory_allocated(),
                    "gpu_memory_reserved": torch.cuda.memory_reserved(),
                    "gpu_memory_free": torch.cuda.memory_reserved() - torch.cuda.memory_allocated(),
                }
            )

        return info

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.reset()
        except Exception:
            # Ignore cleanup errors during destruction
            pass


def create_trainable_predictor(
    predictor: LLMPredictorInterface,
    trainer: LLMTrainerInterface,
    save_adapters: bool = True,
    cleanup_after_training: bool = True,
) -> TrainableLLMPredictor:
    """
    Convenience function to create a trainable predictor with common settings.

    Args:
        predictor: LLM predictor for inference
        trainer: LLM trainer for fine-tuning
        save_adapters: Whether to save trained adapters
        cleanup_after_training: Whether to cleanup after training

    Returns:
        Configured TrainableLLMPredictor instance
    """
    config = TrainableLLMPredictorConfig(
        save_adapters=save_adapters,
        cleanup_after_training=cleanup_after_training,
    )

    return TrainableLLMPredictor(predictor, trainer, config)
