"""Feature presets: 'default' is leak-free; 'paper' (VSI-Bench only) is the proceedings' feature list."""

import logging

import pytest

from TsT.benchmarks.cvb.benchmark import CVBBenchmark
from TsT.benchmarks.vsi.benchmark import VSIBenchmark
from TsT.benchmarks.vsi.models import GOLD_PAIR_FEATURES, RelDistanceModel

# The proceedings' object_rel_distance feature list, in its original order (order matters:
# it changes the Random Forest's feature subsampling).
PAPER_REL_DISTANCE = [
    "object_1",
    "object_2",
    "object_3",
    "object_4",
    "target_object",
    "max_option_freq",
    "max_tgt_option_pair_freq",
    "max_tgt_option_ord_pair_freq",
    "opt_0_option_freq",
    "opt_0_tgt_option_pair_freq",
    "opt_0_tgt_option_ord_pair_freq",
    "opt_1_option_freq",
    "opt_1_tgt_option_pair_freq",
    "opt_1_tgt_option_ord_pair_freq",
    "opt_2_option_freq",
    "opt_2_tgt_option_pair_freq",
    "opt_2_tgt_option_ord_pair_freq",
    "opt_3_option_freq",
    "opt_3_tgt_option_pair_freq",
    "opt_3_tgt_option_ord_pair_freq",
]


def test_gold_pair_features_are_the_ten_documented_columns():
    expected = {f"opt_{i}_tgt_option_pair_freq" for i in range(4)}
    expected |= {f"opt_{i}_tgt_option_ord_pair_freq" for i in range(4)}
    expected |= {"max_tgt_option_pair_freq", "max_tgt_option_ord_pair_freq"}
    assert len(GOLD_PAIR_FEATURES) == 10 and set(GOLD_PAIR_FEATURES) == expected


def test_paper_preset_is_the_proceedings_list():
    assert RelDistanceModel(feature_set="paper").feature_cols == PAPER_REL_DISTANCE


def test_default_preset_drops_exactly_the_gold_pair_columns_in_order():
    assert RelDistanceModel().feature_cols == [c for c in PAPER_REL_DISTANCE if c not in GOLD_PAIR_FEATURES]
    assert RelDistanceModel.feature_cols == RelDistanceModel().feature_cols  # class default is leak-free


def test_presets_differ_only_in_object_rel_distance():
    default = {m.name: m.feature_cols for m in VSIBenchmark().get_feature_based_models("default")}
    paper = {m.name: m.feature_cols for m in VSIBenchmark().get_feature_based_models("paper")}
    assert default.keys() == paper.keys()
    assert [n for n in default if default[n] != paper[n]] == ["object_rel_distance"]


def test_paper_preset_warns_on_every_call(caplog):
    with caplog.at_level(logging.WARNING, logger="TsT"):
        VSIBenchmark().get_feature_based_models("paper")
        VSIBenchmark().get_feature_based_models("paper")
    warnings = [r for r in caplog.records if "held-out" in r.getMessage()]
    assert len(warnings) == 2


def test_default_preset_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="TsT"):
        VSIBenchmark().get_feature_based_models()
    assert not caplog.records


def test_cvb_paper_preset_points_to_the_withdrawal():
    with pytest.raises(ValueError, match=r"75\.5.*withdrawn"):
        CVBBenchmark().get_feature_based_models("paper")


@pytest.mark.parametrize("bench", [VSIBenchmark, CVBBenchmark])
def test_unknown_feature_set_raises(bench):
    with pytest.raises(ValueError, match="Unknown feature set"):
        bench().get_feature_based_models("nope")
    with pytest.raises(ValueError):
        RelDistanceModel(feature_set="nope")
