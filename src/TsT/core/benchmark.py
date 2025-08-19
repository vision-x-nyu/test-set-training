"""
Benchmark registry and base classes for the TsT evaluation framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type
import pandas as pd
from .protocols import FeatureBasedBiasModel, QuestionAnswerBiasModel


class BenchmarkRegistry:
    """Registry for all available benchmarks with auto-discovery."""

    _benchmarks: Dict[str, Type["Benchmark"]] = {}

    @classmethod
    def register(cls, benchmark_class: Type["Benchmark"]) -> Type["Benchmark"]:
        """Decorator to register a benchmark class."""
        cls._benchmarks[benchmark_class.name] = benchmark_class
        return benchmark_class

    @classmethod
    def get_benchmark(cls, name: str) -> "Benchmark":
        """Get a benchmark instance by name, with lazy loading."""
        if name not in cls._benchmarks:
            # Try to import the benchmark to trigger registration
            try:
                import importlib

                importlib.import_module(f"TsT.benchmarks.{name}")
            except ImportError:
                pass

        if name not in cls._benchmarks:
            available = list(cls._benchmarks.keys())
            raise ValueError(f"Unknown benchmark: {name}. Available: {available}")

        return cls._benchmarks[name]()

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmark names."""
        # Try to load all known benchmarks first
        known_benchmarks = ["cvb", "vsi", "video_mme", "mmmu"]
        for benchmark_name in known_benchmarks:
            try:
                import importlib

                importlib.import_module(f"TsT.benchmarks.{benchmark_name}")
            except ImportError:
                continue

        return list(cls._benchmarks.keys())

    @classmethod
    def get_all_benchmarks(cls) -> Dict[str, "Benchmark"]:
        """Get all registered benchmarks as instances."""
        cls.list_benchmarks()  # Trigger lazy loading
        return {name: benchmark_class() for name, benchmark_class in cls._benchmarks.items()}


class Benchmark(ABC):
    """Abstract base class for all benchmarks."""

    name: str | None = None  # Must be set by subclasses
    description: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            raise ValueError(f"Benchmark {cls.__name__} must define a 'name' class attribute")

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load the benchmark dataset."""
        pass

    @abstractmethod
    def get_feature_based_models(self) -> List[FeatureBasedBiasModel]:
        """Get all feature-based models for RF evaluation (one per question type)."""
        pass

    @abstractmethod
    def get_qa_models(self) -> List[QuestionAnswerBiasModel]:
        """Get a single QA model for LLM evaluation of this benchmark."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get benchmark metadata. Override for custom metadata."""
        feature_models = self.get_feature_based_models()
        return {
            "name": self.name,
            "description": self.description,
            "question_types": [model.name for model in feature_models],
            "formats": self._get_format_mapping(feature_models),
            "num_feature_models": len(feature_models),
        }

    def _get_format_mapping(self, models: List[FeatureBasedBiasModel]) -> Dict[str, List[str]]:
        """Group question types by format."""
        formats = {}
        for model in models:
            format_key = model.format
            if format_key not in formats:
                formats[format_key] = []
            formats[format_key].append(model.name)
        return formats
