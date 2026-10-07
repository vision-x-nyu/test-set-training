"""Test-set Stress-Test (TsT): expose non-visual shortcuts in multimodal benchmarks.

Heavy dependencies (pandas, scikit-learn, pyarrow, huggingface_hub) are imported on first use, so
``import TsT`` and ``python -m TsT --help`` stay fast.
"""

__version__ = "1.0.0"

_LAZY = {
    "run_evaluation": ("evaluation", "run_evaluation"),
    "evaluate_benchmark": ("evaluation", "evaluate_benchmark"),
    "BenchmarkRun": ("evaluation", "BenchmarkRun"),
    "FeatureBasedBiasModel": ("core.protocols", "FeatureBasedBiasModel"),
    "Benchmark": ("core.benchmark", "Benchmark"),
    "BenchmarkRegistry": ("core.benchmark", "BenchmarkRegistry"),
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(f".{module_name}", __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
