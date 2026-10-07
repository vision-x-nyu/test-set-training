"""Data loading, scoring and aggregation helpers shared by the TsT evaluators."""

import hashlib
import numbers
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@lru_cache(maxsize=1)
def get_t2d():
    """Shared text2digits converter (https://github.com/ShailChoksi/text2digits)."""
    from text2digits import text2digits

    return text2digits.Text2Digits()


# =============================================================================
# STRING EVALUATION ----------------------------------------------------------
# =============================================================================


def exact_match(pred, target):
    if isinstance(pred, str):
        return 1.0 if pred.lower() == target.lower() else 0.0
    else:
        return 1.0 if float(pred) == float(target) else 0.0


def fuzzy_cleanup(pred: str) -> str:
    return str(pred).strip().split(" ")[0].rstrip(".").strip().lower()


def fuzzy_match(pred, target):
    cleaned_pred = fuzzy_cleanup(pred.lower())
    cleaned_target = fuzzy_cleanup(target.lower())
    return float(cleaned_pred == cleaned_target)


def is_integer(value: Any) -> bool:
    """
    True if value represents an integer:
      - int (but not bool) -> True
      - float that is numerically an integer (e.g., 5.0, -3.0) -> True
      - str of a base-10 int (optional +/- and whitespace) -> True
      - otherwise -> False
    """
    # Avoid treating True/False as integers
    if isinstance(value, bool):
        return False

    # Native integers (and numpy integer types, etc.)
    if isinstance(value, numbers.Integral):
        return True

    # Floats: only True if they’re exactly an integer value
    if isinstance(value, float):
        return value.is_integer()

    # Strings: accept optional sign and whitespace; digits only
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return False
        if s[0] in "+-":
            s = s[1:]
        return s.isdigit()

    return False


# =============================================================================
# NUMERIC EVALUATION ----------------------------------------------------------
# =============================================================================


def fuzzy_cleanup_numeric(pred: str) -> float:
    t2d = get_t2d()
    cleaned_pred = pred.strip().lower().rstrip(".").strip()
    is_negative = False
    # Handle 'minus' or 'negative' at the start
    for neg_word in ("minus ", "negative "):
        if cleaned_pred.startswith(neg_word):
            is_negative = True
            cleaned_pred = cleaned_pred[len(neg_word) :].strip()
            break

    try:
        # Try text2digits conversion first (for text numbers like "forty two")
        # Only strip currency symbols to avoid breaking valid numbers
        text2digits_input = re.sub(r"[€£$]", "", cleaned_pred)
        converted_pred = t2d.convert(text2digits_input)
    except Exception:
        # If text2digits fails, fall back to using the original string
        converted_pred = cleaned_pred

    # Extract the first numeric pattern (supports commas and decimals)
    match = re.search(r"-?\d[\d,]*\.?\d*", converted_pred)
    if not match:
        # raise ValueError(f"No numeric value found in '{pred}'")
        # NOTE: an arbitrary model may return a non-numeric value. o/p NaN instead of raising an error
        return np.nan

    numeric_str = match.group(0).replace(",", "")
    result = float(numeric_str)
    # Apply negation if original input started with 'minus' or 'negative'
    if is_negative and not numeric_str.startswith("-"):
        result = -result
    return result


def mean_relative_accuracy(pred, target, start: float = 0.5, end: float = 0.95, step: float = 0.05) -> float:
    """Compute mean relative accuracy across thresholds.

    MRA averages, over the thresholds t in {start, start + step, ..., end}, whether
    the relative error |pred - target| / |target| is strictly below 1 - t.
    NOTE: the comparison is strict (<). The official VSI-Bench scorer uses <=; the
    two differ only when the relative error lands exactly on a threshold.

    Handles division-by-zero cases gracefully by defining:
    - If target == 0 and pred == 0 -> relative error = 0
    - If target == 0 and pred != 0 -> relative error = inf
    """

    # Convert inputs to numpy arrays (supports scalars and arrays) with broadcasting
    pred_arr = np.asarray(pred, dtype=float)
    target_arr = np.asarray(target, dtype=float)

    # Check for NaN values in the arrays
    if np.any(np.isnan(pred_arr)) or np.any(np.isnan(target_arr)):
        return np.nan

    thresholds = np.linspace(start, end, int((end - start) / step) + 2)

    pred_b, target_b = np.broadcast_arrays(pred_arr, target_arr)
    diff = np.abs(pred_b - target_b)

    # Initialize relative error as infinity by default
    rel_err = np.full_like(diff, np.inf, dtype=float)

    # Compute where target is non-zero
    nonzero_mask = target_b != 0
    rel_err[nonzero_mask] = diff[nonzero_mask] / np.abs(target_b[nonzero_mask])

    # Define 0/0 as 0 relative error
    zero_zero_mask = (~nonzero_mask) & (np.abs(pred_b) == 0)
    rel_err[zero_zero_mask] = 0.0

    return float(np.mean([(rel_err < 1 - t).mean() for t in thresholds]))


def fuzzy_mra(pred, target, start=0.5, end=0.95, step=0.05) -> float:
    cleaned_pred = fuzzy_cleanup_numeric(pred)
    cleaned_target = fuzzy_cleanup_numeric(target)
    return mean_relative_accuracy(cleaned_pred, cleaned_target, start, end, step)


def weighted_mean_std(scores: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """
    Weighted mean:
        wgt_mean = sum(score_i * count_i) / sum(count_i)
    Weighted variance:
        weighted_var = sum(count_i * (score_i - wgt_mean)**2) / sum(count_i)
    Weighted std:
        weighted_std = sqrt(weighted_var)
    Only models with count > 0 are included.
    """
    scores = np.array(scores)
    counts = np.array(counts)

    mask = counts > 0
    scores = scores[mask]
    counts = counts[mask]
    wgt_mean = (scores * counts).sum() / counts.sum() if counts.sum() > 0 else 0
    wgt_var = ((counts * (scores - wgt_mean) ** 2).sum() / counts.sum()) if counts.sum() > 0 else 0
    wgt_std = wgt_var**0.5
    return wgt_mean, wgt_std


# =============================================================================
# CONFIDENCE INTERVALS --------------------------------------------------------
# =============================================================================


def wilson_ci(num_correct: int, num_samples: int, alpha: float = 0.05) -> tuple[float, tuple[float, float]]:
    """Wilson CI (no continuity correction) on micro accuracy

    NOTE: only applicable to binary classification (e.g., accuracy)

    Args:
        num_correct: number correct from ONE out-of-fold prediction per sample (possibly after averaging probs across repeats)
        num_samples: total samples
        alpha: significance level (default: 0.05)

    Returns:
        p: proportion correct
        ci: tuple of lower and upper bounds of the Wilson CI
    """
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p = num_correct / num_samples  # proportion correct
    denom = 1 + z**2 / num_samples
    center = (p + z**2 / (2 * num_samples)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / num_samples + z**2 / (4 * num_samples**2))
    ci = (center - half, center + half)
    return p, ci


def bootstrap_mean_ci(values: Iterable[float], B=10_000, alpha=0.05, seed=0) -> tuple[float, tuple[float, float]]:
    """Two-sided bootstrap mean confidence interval

    NOTE: applicable to any metric

    Args:
        values: array of values
        B: number of bootstrap samples (default: 10_000)
        alpha: significance level (default: 0.05)
        seed: random seed (default: 0)

    Returns:
        m: mean of values
        ci: tuple of lower and upper bounds of the bootstrap confidence interval
    """
    from tqdm import trange

    rng = np.random.default_rng(seed)
    values = np.array(values)
    n = len(values)
    boots = np.empty(B)  # store bootstrap sample means

    # generate B bootstrap samples by resampling w/ replacement
    for b in trange(B, desc="Bootstrap CI"):
        idx = rng.integers(0, n, size=n)  # n random indices w/ replacement
        boots[b] = values[idx].mean()  # mean of bootstrap sample

    # compute percentile-based confidence interval
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return values.mean(), (lo, hi)


def stratified_bootstrap_mean_ci(
    groups: Iterable[Any], values: Iterable[float], B: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    """Stratified two-sided bootstrap mean confidence interval

    NOTE: applicable to any metric

    Args:
        groups: array of group labels. Shape: (n,)
        values: array of values. Shape: (n,)
        B: number of bootstrap samples (default: 10_000)
        alpha: significance level (default: 0.05)
        seed: random seed (default: 0)

    Returns:
        m: mean of values
        ci: tuple of lower and upper bounds of the stratified bootstrap confidence interval
    """
    from tqdm import trange

    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    values = np.asarray(values)

    assert len(groups) == len(values), (
        f"groups and values must have the same length, got {len(groups)} and {len(values)}"
    )

    uniq = np.unique(groups)
    boots = np.empty(B)

    for b in trange(B, desc=f"Stratified Bootstrap CI ({len(uniq)} groups)"):
        parts = []
        # resample within each group (stratum)
        for g in uniq:
            idx = np.where(groups == g)[0]  # indices for this group
            bs = rng.choice(idx, size=len(idx), replace=True)  # n random indices w/ replacement
            parts.append(values[bs])
        boots[b] = np.concatenate(parts).mean()  # combine groups and compute mean

    # compute confidence interval from bootstrap distribution
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return values.mean(), (lo, hi)


# =============================================================================
# DATA LOADING ----------------------------------------------------------------
# =============================================================================


def load_hf_split(
    repo_id: str, revision: str, pattern: str = "test*.parquet", drop_columns: Iterable[str] = ()
) -> "pd.DataFrame":
    """One split of a Hugging Face dataset repo at an exact revision, as a DataFrame.

    Fetches the parquet files matching ``pattern`` with huggingface_hub and reads them in
    sorted file-name order, which is the order the ``datasets`` library uses for the same
    glob. huggingface_hub caches files per revision, so an offline run (HF_HUB_OFFLINE=1)
    or a run with the Hub unreachable reads the requested revision from the cache or
    fails; it never falls back to another cached revision. ``drop_columns`` are skipped
    when reading (e.g. embedded images).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    kwargs = dict(repo_type="dataset", revision=revision, allow_patterns=[pattern])
    try:
        local = snapshot_download(repo_id, **kwargs)
    except (RepositoryNotFoundError, RevisionNotFoundError):
        raise
    except Exception as hub_error:  # offline or Hub unreachable: use this revision's cached snapshot, if any
        try:
            local = snapshot_download(repo_id, local_files_only=True, **kwargs)
        except Exception:
            raise hub_error from None
    files = sorted(Path(local).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {repo_id} at revision {revision} ({local})")
    drop = set(drop_columns)
    tables = []
    for f in files:
        columns = [c for c in pq.read_schema(f).names if c not in drop]
        tables.append(pq.read_table(f, columns=columns))
    return pa.concat_tables(tables).to_pandas()


def row_order_sha1(df: "pd.DataFrame", id_col: str) -> str:
    """SHA-1 fingerprint of the row order: the ids in ``id_col``, comma-joined in row order.

    Shuffled k-fold assignment depends on row order, so two loads of a benchmark give the
    same TsT-RF scores only if their fingerprints match.
    """
    return hashlib.sha1(",".join(map(str, df[id_col].tolist())).encode()).hexdigest()
