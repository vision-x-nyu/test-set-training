"""TsT-RF golden values at the pinned HF revisions (downloads the benchmarks; run with -m network).

Expected values are the independent reference runs of the same CPU recipe (5 folds,
seed 42, count-weighted mean over question types; macro = unweighted mean). CV-Bench
downloads about 405 MB of image parquet files on first use.
"""

import os
import subprocess
import sys

import pytest

from TsT.evaluation import evaluate_benchmark
from TsT.utils import row_order_sha1

pytestmark = [pytest.mark.network, pytest.mark.slow]

VSI_BC96B17 = "bc96b17cb6be84878a6c1f2e64c24346356e0d04"
VSI_D7CB1A3 = "d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d"
CVB_BC284DB = "bc284db50d036958861cb60cdd7b77612052ce0d"

# (benchmark, revision, feature_set, group_col) -> (weighted, macro); None = not pinned
GOLDENS = {
    ("vsi", VSI_BC96B17, "default", None): (43.00, None),
    ("vsi", VSI_BC96B17, "default", "scene_name"): (38.82, None),
    ("vsi", VSI_BC96B17, "paper", None): (43.50, 42.17),
    ("vsi", VSI_BC96B17, "paper", "scene_name"): (38.98, 37.33),
    # same questions in the post-2025-11-11 row order: different folds, slightly different scores
    ("vsi", VSI_D7CB1A3, "paper", None): (43.64, 42.38),
    ("vsi", VSI_D7CB1A3, "paper", "scene_name"): (38.73, 37.21),
    ("cvb", CVB_BC284DB, "default", None): (56.14, 55.32),
    ("cvb", CVB_BC284DB, "default", "image_id"): (55.65, 54.65),
}

# row_order_sha1 of each revision's test split (over "id" for VSI-Bench, "idx" for CV-Bench)
FINGERPRINTS = {
    ("vsi", VSI_BC96B17): "8c48d42738671ef11a1ccb6300c81ed5d893f10f",
    ("vsi", VSI_D7CB1A3): "b48270fe65bfe142baa12835a230797f6e4f4ec7",
    ("cvb", CVB_BC284DB): "02535f1c609db8bba7c6bd64d9a78b9aa3d0fa3a",
}

_DATA = {}


def _data(benchmark, revision):
    from TsT.core.benchmark import BenchmarkRegistry

    if (benchmark, revision) not in _DATA:
        _DATA[(benchmark, revision)] = BenchmarkRegistry.get_benchmark(benchmark).load_data(revision=revision)
    return _DATA[(benchmark, revision)]


@pytest.mark.parametrize("key", list(GOLDENS), ids=lambda k: f"{k[0]}-{k[1][:7]}-{k[2]}-{k[3] or 'random'}")
def test_rf_golden(key):
    benchmark, revision, feature_set, group_col = key
    weighted, macro = GOLDENS[key]
    run = evaluate_benchmark(
        benchmark,
        revision=revision,
        feature_set=feature_set,
        group_col=group_col,
        show_progress=False,
        df=_data(benchmark, revision),
    )
    assert not run.errors
    assert row_order_sha1(run.df, run.id_col) == FINGERPRINTS[(benchmark, revision)]
    assert 100 * run.weighted_mean == pytest.approx(weighted, abs=0.005)
    if macro is not None:
        assert 100 * run.macro_mean == pytest.approx(macro, abs=0.005)


def test_default_revisions_are_the_pinned_ones():
    from TsT.benchmarks.cvb.benchmark import CVBBenchmark
    from TsT.benchmarks.vsi.benchmark import VSIBenchmark

    assert VSIBenchmark.default_revision == VSI_BC96B17
    assert CVBBenchmark.default_revision == CVB_BC284DB


def test_vsi_s_x_joins_hf_ids_one_to_one():
    run = evaluate_benchmark("vsi", show_progress=False, df=_data("vsi", VSI_BC96B17))
    s_x = run.s_x()
    assert len(s_x) == 5130 and s_x["id"].is_unique
    assert set(s_x["id"]) == set(run.df["id"])


def test_offline_run_reads_the_pinned_revision_with_others_cached():
    """With several VSI-Bench revisions cached, an offline load still reads the pinned one."""
    for revision in (VSI_BC96B17, VSI_D7CB1A3):  # make sure both are cached; the 2025-11-11 one last
        _DATA.pop(("vsi", revision), None)
        _data("vsi", revision)
    code = (
        "from TsT.benchmarks.vsi.benchmark import VSIBenchmark; from TsT.utils import row_order_sha1; "
        "df = VSIBenchmark().load_data(); print(len(df), row_order_sha1(df, 'id'))"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.split() == ["5130", FINGERPRINTS[("vsi", VSI_BC96B17)]]
