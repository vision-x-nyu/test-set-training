"""
Abstract interfaces for LLM trainers.

This module defines the base interfaces that all LLM trainers must implement,
following DataEnvGym patterns for clean separation of training concerns.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict
from ..data.models import TrainingDatum, LoRAAdapterInfo, TrainingProgress


class LLMTrainerInterface(ABC):
    """Abstract interface for LLM trainers"""

    @abstractmethod
    def train(self, training_data: List[TrainingDatum], output_dir: Path) -> LoRAAdapterInfo:
        """
        Train LoRA adapter and return adapter info.

        Args:
            training_data: List of training examples
            output_dir: Directory to save the trained adapter

        Returns:
            LoRAAdapterInfo with details about the trained adapter
        """
        pass

    @abstractmethod
    def validate_training_data(self, training_data: List[TrainingDatum]) -> bool:
        """
        Validate that training data is suitable for training.

        Args:
            training_data: Training data to validate

        Returns:
            True if data is valid, raises ValueError otherwise
        """
        pass

    @property
    @abstractmethod
    def is_training(self) -> bool:
        """Check if training is currently in progress"""
        pass

    @abstractmethod
    def stop_training(self) -> None:
        """Stop training if currently in progress"""
        pass


class ProgressCallback:
    """Callback interface for training progress updates"""

    def on_training_start(self, total_steps: int) -> None:
        """Called when training starts"""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch"""
        pass

    def on_step(self, progress: TrainingProgress) -> None:
        """Called after each training step"""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch"""
        pass

    def on_training_end(self, final_metrics: Dict[str, float]) -> None:
        """Called when training completes"""
        pass


class BaseLLMTrainer(LLMTrainerInterface):
    """
    Base implementation with common functionality for LLM trainers.

    This provides default implementations and utilities that concrete
    trainers can inherit and extend.
    """

    def __init__(self, progress_callback: Optional[ProgressCallback] = None):
        self._is_training = False
        self._progress_callback = progress_callback
        self._should_stop = False

    @property
    def is_training(self) -> bool:
        """Check if training is currently in progress"""
        return self._is_training

    def stop_training(self) -> None:
        """Stop training if currently in progress"""
        self._should_stop = True

    def validate_training_data(self, training_data: List[TrainingDatum]) -> bool:
        """
        Validate that training data is suitable for training.

        Args:
            training_data: Training data to validate

        Returns:
            True if data is valid, raises ValueError otherwise
        """
        if not training_data:
            raise ValueError("No training data provided")

        if len(training_data) < 2:
            raise ValueError("At least 2 training examples required")

        # Check for valid instructions and responses
        for i, datum in enumerate(training_data):
            if not datum.instruction.strip():
                raise ValueError(f"Empty instruction in training datum {i}")

            if not datum.response.strip():
                raise ValueError(f"Empty response in training datum {i}")

        return True

    def _set_training_state(self, is_training: bool) -> None:
        """Set training state (for use by subclasses)"""
        self._is_training = is_training
        if not is_training:
            self._should_stop = False

    def _should_stop_training(self) -> bool:
        """Check if training should be stopped"""
        return self._should_stop

    def _notify_progress(self, progress: TrainingProgress) -> None:
        """Notify progress callback if available"""
        if self._progress_callback:
            self._progress_callback.on_step(progress)

    def _notify_training_start(self, total_steps: int) -> None:
        """Notify training start"""
        if self._progress_callback:
            self._progress_callback.on_training_start(total_steps)

    def _notify_training_end(self, final_metrics: Dict[str, float]) -> None:
        """Notify training end"""
        if self._progress_callback:
            self._progress_callback.on_training_end(final_metrics)

    def _prepare_output_directory(self, output_dir: Path) -> Path:
        """Prepare output directory for training"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organization
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)

        return output_dir
