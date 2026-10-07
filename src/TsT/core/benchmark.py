"""
Benchmark registry and base classes for the TsT evaluation framework.
"""

from __future__ import annotations

import importlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from ..utils import row_order_sha1
from .protocols import FeatureBasedBiasModel, IBPStrategy, QuestionAnswerBiasModel

logger = logging.getLogger(__name__)

# Benchmarks that ship with the package (imported on demand to trigger registration).
BUILTIN_BENCHMARKS: Tuple[str, ...] = ("vsi", "cvb", "mmmu", "video_mme", "mmstar")


class DatasetRevisionError(RuntimeError):
    """The loaded benchmark rows do not match the requested dataset revision."""


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
        if name not in cls._benchmarks and name in BUILTIN_BENCHMARKS:
            importlib.import_module(f"TsT.benchmarks.{name}")

        if name not in cls._benchmarks:
            available = sorted(set(cls._benchmarks) | set(BUILTIN_BENCHMARKS))
            raise ValueError(f"Unknown benchmark: {name}. Available: {', '.join(available)}")

        return cls._benchmarks[name]()

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmark names (importing the built-in ones first)."""
        for benchmark_name in BUILTIN_BENCHMARKS:
            importlib.import_module(f"TsT.benchmarks.{benchmark_name}")
        return list(cls._benchmarks.keys())

    @classmethod
    def get_all_benchmarks(cls) -> Dict[str, "Benchmark"]:
        """Get all registered benchmarks as instances."""
        cls.list_benchmarks()  # Trigger lazy loading
        return {name: benchmark_class() for name, benchmark_class in cls._benchmarks.items()}


class Benchmark(ABC):
    """Abstract base class for all benchmarks.

    A benchmark implements ``load_data`` plus ``get_feature_based_models`` (TsT-RF),
    ``get_qa_models`` (TsT-LLM), or both, and optionally ``get_ibp_strategy``. The
    class attributes below describe where the data come from and how results should
    be reported.
    """

    name: str | None = None  # Must be set by subclasses
    description: str = ""

    # Hugging Face dataset repo and the revision (commit sha) that load_data pins by default.
    hf_repo: Optional[str] = None
    default_revision: Optional[str] = None
    # Revision pinned for TsT-LLM runs when it differs from default_revision (VSI-Bench: the
    # paper's TsT-LLM runs used a later revision than its TsT-RF runs). None = default_revision.
    llm_revision: Optional[str] = None
    # Column holding the benchmark's own question id (used for per-question exports).
    id_col: str = "id"
    # Suggested --group_col for grouped folds (e.g. the scene or image a question is about).
    group_col_hint: Optional[str] = None
    # Named feature presets accepted by get_feature_based_models().
    feature_sets: Tuple[str, ...] = ("default",)
    # Known row-order fingerprints (TsT.utils.row_order_sha1 over id_col), keyed by full
    # revision sha. load_data checks every load against these; see check_row_order.
    row_order_fingerprints: Dict[str, str] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            raise ValueError(f"Benchmark {cls.__name__} must define a 'name' class attribute")

    @abstractmethod
    def load_data(self, revision: Optional[str] = None) -> pd.DataFrame:
        """Load the benchmark dataset (``revision=None`` means ``default_revision``)."""
        pass

    def get_feature_based_models(self, feature_set: str = "default") -> List[FeatureBasedBiasModel]:
        """Feature-based models for TsT-RF (one per question type)."""
        raise NotImplementedError(f"TsT-RF is not available for benchmark '{self.name}'; use --mode llm.")

    @property
    def supports_rf(self) -> bool:
        return type(self).get_feature_based_models is not Benchmark.get_feature_based_models

    @property
    def supports_llm(self) -> bool:
        return type(self).get_qa_models is not Benchmark.get_qa_models

    def pinned_revision(self, mode: str = "rf") -> Optional[str]:
        """The revision a run of ``mode`` ("rf" or "llm") loads by default."""
        if mode == "llm" and self.llm_revision is not None:
            return self.llm_revision
        return self.default_revision

    def resolve_revision(self, revision: Optional[str] = None, mode: str = "rf") -> Optional[str]:
        """``None`` means the pinned revision for ``mode``; an abbreviation (7+ hex digits) of a
        pinned or fingerprinted revision expands to the full sha."""
        pinned = self.pinned_revision(mode)
        if revision is None:
            return pinned
        rev = revision.strip()
        if re.fullmatch(r"[0-9a-f]{7,40}", rev):
            known = [pinned, self.default_revision, self.llm_revision, *self.row_order_fingerprints]
            for full in known:
                if full and full.startswith(rev):
                    return full
        return rev

    def check_row_order(self, df: pd.DataFrame, revision: Optional[str]) -> str:
        """Check freshly loaded rows against the known fingerprint of ``revision``.

        Raises DatasetRevisionError if ``revision`` has a known fingerprint and the rows
        do not match it. For any revision other than ``default_revision``, logs a warning
        with the fingerprint that was loaded. Returns the loaded fingerprint.
        """
        fingerprint = row_order_sha1(df, self.id_col)
        expected = self.row_order_fingerprints.get(revision or "")
        if expected is not None and fingerprint != expected:
            raise DatasetRevisionError(
                f"The {self.hf_repo} rows that were loaded do not match revision {revision}: row-order "
                f"fingerprint {fingerprint} ({len(df)} rows), expected {expected}. A different version of the "
                "dataset was read, for example a cached copy of another revision used because the Hugging Face "
                "Hub was unreachable or HF_HUB_OFFLINE=1 was set. Scores computed on these rows would not match "
                "the published values. Delete this dataset from the Hugging Face cache (under HF_HOME) and run "
                "again with network access."
            )
        if revision not in (self.default_revision, self.llm_revision):
            logger.warning(
                f"{self.hf_repo} @ {revision} is not a pinned revision ({self.default_revision}). Loaded "
                f"{len(df)} rows with row-order fingerprint {fingerprint}"
                + (" (the known fingerprint of this revision)" if expected is not None else "")
                + ". Shuffled folds depend on row order, so scores can differ from the published values."
            )
        return fingerprint

    def check_feature_set(self, feature_set: str) -> None:
        """Raise ValueError if ``feature_set`` is not one of this benchmark's presets."""
        if feature_set not in self.feature_sets:
            raise ValueError(
                f"Unknown feature set '{feature_set}' for benchmark '{self.name}'. Available: {list(self.feature_sets)}"
            )

    def get_qa_models(self) -> List[QuestionAnswerBiasModel]:
        """Question-answer models for TsT-LLM (one per answer format)."""
        raise NotImplementedError(f"TsT-LLM is not available for benchmark '{self.name}'.")

    def get_ibp_strategy(self) -> IBPStrategy:
        """Selection strategy for Iterative Bias Pruning (IBP) on this benchmark."""
        raise NotImplementedError(f"No IBP strategy is defined for benchmark '{self.name}'.")

    def get_metadata(self) -> Dict[str, Any]:
        """Get benchmark metadata. Override for custom metadata."""
        meta = {
            "name": self.name,
            "description": self.description,
            "hf_repo": self.hf_repo,
            "default_revision": self.default_revision,
            "llm_revision": self.pinned_revision("llm"),
            "row_order_fingerprints": dict(self.row_order_fingerprints),
            "modes": [m for m, ok in (("rf", self.supports_rf), ("llm", self.supports_llm)) if ok],
        }
        if self.supports_rf:
            feature_models = self.get_feature_based_models()
            meta.update(
                {
                    "feature_sets": list(self.feature_sets),
                    "question_types": [model.name for model in feature_models],
                    "formats": self._get_format_mapping(feature_models),
                    "num_feature_models": len(feature_models),
                }
            )
        return meta

    def _get_format_mapping(self, models: List[FeatureBasedBiasModel]) -> Dict[str, List[str]]:
        """Group question types by format."""
        formats = {}
        for model in models:
            formats.setdefault(model.format, []).append(model.name)
        return formats
