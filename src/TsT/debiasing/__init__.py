"""
Iterative Bias Pruning (IBP): remove the questions whose TsT scores show the most exploitable
non-visual signal, re-running TsT after every batch.

Command line: ``python -m TsT.debiasing --help``. Python::

    from TsT.debiasing import debias_benchmark
    run = debias_benchmark("vsi", mode="rf", alloc="per_format", budget=1000)
    run.save("outputs/ibp_vsi_b1000")   # removed_ids.txt, kept_ids.txt, summary.json

Submodules load on first use, so importing this package stays cheap.
"""

_LAZY = {
    "IBPConfig": ("config", "IBPConfig"),
    "TopKDiversityStrategy": ("strategies", "TopKDiversityStrategy"),
    "MissingScoreError": ("strategies", "MissingScoreError"),
    "Group": ("allocation", "Group"),
    "plan_groups": ("allocation", "plan_groups"),
    "proportional_budgets": ("allocation", "proportional_budgets"),
    "uniform_budgets": ("allocation", "uniform_budgets"),
    "budgets_from_ids": ("allocation", "budgets_from_ids"),
    "IBPIterationResult": ("results", "IBPIterationResult"),
    "IBPResult": ("results", "IBPResult"),
    "GroupResult": ("results", "GroupResult"),
    "IBPRun": ("results", "IBPRun"),
    "aggregate_sample_predictions": ("ibp", "aggregate_sample_predictions"),
    "iterative_bias_pruning": ("ibp", "iterative_bias_pruning"),
    "debias_benchmark": ("ibp", "debias_benchmark"),
    "read_id_list": ("ibp", "read_id_list"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(f".{module_name}", __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
