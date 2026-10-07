"""Pin the unit-aware NUM parser behind Table 12's InternVL rows (reproduce/robust_num.py)."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "reproduce" / "robust_num.py"
_spec = importlib.util.spec_from_file_location("robust_num", _PATH)
rn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rn)


@pytest.mark.parametrize(
    "pred,qt,expected",
    [
        ("1.5 meters.", "object_abs_distance", 1.5),
        ("1.5 meters.", "object_size_estimation", 150.0),  # sizes are in cm
        ("90 centimeters.", "object_abs_distance", 0.9),  # distances are in m
        ("Two.", "object_counting", 2.0),
        ("twenty five", "object_counting", 25.0),
        ("30 square meters.", "room_size_estimation", 30.0),
        ("about 3 to 4 ft", "object_abs_distance", 3.0),  # first number; feet are not converted
        ("50 millimeters", "object_size_estimation", 5000.0),  # kept quirk: "millimeter" matches the meter rule
        ("2", "object_counting", 2.0),
    ],
)
def test_robust_num(pred, qt, expected):
    assert rn.robust_num(pred, qt) == pytest.approx(expected)


def test_no_number_scores_zero():
    assert rn.robust_num("I cannot tell.", "object_counting") is None
    assert rn.robust_mra("I cannot tell.", "3", "object_counting") == 0.0


def test_robust_mra_matches_lmms_eval_mra():
    assert rn.robust_mra("1.5 meters.", "1.5", "object_abs_distance") == 1.0
    # relative error 0.27 passes the thresholds t = 0.50, ..., 0.70 (0.27 <= 1 - t): 5 of 10
    assert rn.robust_mra("1.27 m", "1.0", "object_abs_distance") == pytest.approx(0.5)
    assert rn.robust_mra("3", "0", "object_counting") == 0.0
