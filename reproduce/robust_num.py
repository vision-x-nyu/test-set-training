"""Unit-aware parsing of numerical (NUM) VSI-Bench answers, used for every NUM score in reproduce/.

The shipped prediction records carry a per-sample MRA field from our evaluation fork, which
parsed the whole answer with a bare float(), so "0.6.", "1.5 meters." and "Two." scored 0.
(lmms-eval's own vsibench takes the first word and drops a trailing period; newer versions also
parse number words.) Every NUM answer, vision and blind, is re-parsed with `robust_num` and
scored with `mean_relative_accuracy`:

  1. number words become digits (text2digits: "two" -> "2");
  2. the first number in the answer is taken (trailing units and words are dropped);
  3. lengths are converted to the unit the question asks for: meters for
     `object_abs_distance`, centimeters for `object_size_estimation`. Room sizes (m^2) and
     counts are left as parsed.

The rules are kept exactly as they produced the paper's numbers, including one quirk: in
`object_size_estimation`, "millimeter" also matches the meter test, so a millimeter answer
is multiplied by 100 rather than divided by 10.

Requires numpy and text2digits (a core dependency of the package; tested with 0.1.0).
"""

from __future__ import annotations

import re

import numpy as np

try:
    from text2digits import text2digits as _t2d_module
except ImportError as e:  # without it, number words would silently score 0
    raise ImportError("reproduce/ needs text2digits for NUM scoring: pip install text2digits") from e

NUMRE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_T2D = None


def _text2digits(s):
    global _T2D
    if _T2D is None:
        _T2D = _t2d_module.Text2Digits()
    return _T2D.convert(s)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def robust_num(pred, question_type):
    """Parse a NUM answer to a float in the question's unit; None if it holds no number."""
    s = str(pred).strip().lower()
    try:
        s2 = _text2digits(s.rstrip(".").strip())
    except Exception:
        s2 = s
    m = NUMRE.search(s2.replace(",", ""))
    if not m:
        return None
    try:
        v = float(m.group())
    except Exception:
        return None
    has_cm = ("centimeter" in s) or (re.search(r"\bcm\b", s) is not None)
    has_mm = ("millimeter" in s) or (re.search(r"\bmm\b", s) is not None)
    has_m = (not has_cm) and (re.search(r"meter", s) is not None or re.search(r"\bm\b", s) is not None)
    if question_type == "object_abs_distance":  # meters
        if has_cm:
            v /= 100.0
        elif has_mm:
            v /= 1000.0
    elif question_type == "object_size_estimation":  # centimeters
        if has_m:
            v *= 100.0
        elif has_mm:
            v /= 10.0
    return v


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    """lmms-eval's MRA: share of thresholds c in {0.50, ..., 0.95} with |pred - target| / |target| <= 1 - c."""
    if pred is None or target is None or target == 0:
        return 0.0
    confs = np.linspace(start, end, int((end - start) / interval + 2))
    return float((abs(pred - target) / abs(target) <= 1 - confs).mean())


def robust_mra(prediction, ground_truth, question_type):
    """MRA of one NUM answer after `robust_num` parsing (the ground truth is a plain number)."""
    return mean_relative_accuracy(robust_num(prediction, question_type), to_float(ground_truth))
