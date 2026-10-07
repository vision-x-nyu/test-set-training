"""Leakage canary: a TsT-RF feature of a question must not depend on that question's own answer.

Fit the feature maps once, then recompute features after changing ONLY the answers
(MC: a different valid letter; NUM: another value of the same question type). Any
column in model.feature_cols that changes reads the question's own label, which a
held-out question would leak into its own prediction.
"""

import numpy as np
import pytest

from TsT.benchmarks.cvb.benchmark import CVBBenchmark
from TsT.benchmarks.vsi.benchmark import VSIBenchmark
from TsT.benchmarks.vsi.models import GOLD_PAIR_FEATURES, RelDistanceModel


def _vsi_perturb(raw, rng):
    raw = raw.copy()
    mc = raw["options"].notna()
    raw.loc[mc, "ground_truth"] = raw[mc].apply(
        lambda r: "ABCD"[
            ("ABCD".index(r["ground_truth"]) + 1 + rng.randint(len(r["options"]) - 1)) % len(r["options"])
        ],
        axis=1,
    )
    for _, g in raw[~mc].groupby("question_type"):
        raw.loc[g.index, "ground_truth"] = rng.permutation(g["ground_truth"].values)
    return raw


def _cvb_perturb(raw, rng):
    raw = raw.copy()
    raw["answer"] = raw.apply(
        lambda r: f"({chr(65 + (ord(r['answer'][1]) - 65 + 1 + rng.randint(len(r['choices']) - 1)) % len(r['choices']))})",
        axis=1,
    )
    return raw


BENCH = {  # benchmark class, label perturbation, id column, raw label column
    "vsi": (VSIBenchmark, _vsi_perturb, "id", "ground_truth"),
    "cvb": (CVBBenchmark, _cvb_perturb, "idx", "answer"),
}


def _leaky_columns(bench_name, model, raw):
    bench_cls, perturb, id_col, label_col = BENCH[bench_name]
    raw_perturbed = perturb(raw, np.random.RandomState(0))
    assert (raw_perturbed[label_col] != raw[label_col]).mean() > 0.5
    original = bench_cls.preprocess(raw)
    perturbed = bench_cls.preprocess(raw_perturbed)

    model.fit_feature_maps(model.select_rows(original))
    a = model.add_features(model.select_rows(original)).set_index(id_col)
    b = model.add_features(model.select_rows(perturbed)).set_index(id_col)
    assert len(a) > 0 and a.index.equals(b.index)
    # NaN-safe: equal NaNs count as equal (they do not after astype(str) in pandas >= 3)
    return [c for c in model.feature_cols if not a[c].astype(object).equals(b[c].astype(object))]


def _default_models():
    for m in VSIBenchmark().get_feature_based_models():
        yield pytest.param("vsi", m, id=f"vsi-{m.name}")
    for m in CVBBenchmark().get_feature_based_models():
        yield pytest.param("cvb", m, id=f"cvb-{m.name}")


@pytest.mark.parametrize("bench_name,model", list(_default_models()))
def test_default_features_do_not_read_the_questions_own_label(bench_name, model, request):
    raw = request.getfixturevalue(f"{bench_name}_raw")
    leaky = _leaky_columns(bench_name, model, raw)
    assert leaky == [], f"{model.name}: features keyed on the question's own label: {leaky}"


def test_canary_flags_exactly_the_paper_preset_gold_pair_features(vsi_raw):
    """Positive control: the paper preset's object_rel_distance model reads the label in 10 columns."""
    leaky = _leaky_columns("vsi", RelDistanceModel(feature_set="paper"), vsi_raw)
    assert sorted(leaky) == sorted(GOLD_PAIR_FEATURES)
