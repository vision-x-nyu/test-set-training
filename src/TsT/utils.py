from functools import lru_cache
import re
import numpy as np

# https://github.com/ShailChoksi/text2digits
from text2digits import text2digits


@lru_cache(maxsize=1)
def get_t2d() -> text2digits.Text2Digits:
    return text2digits.Text2Digits()


def fuzzy_cleanup(pred: str) -> str:
    return str(pred).strip().split(" ")[0].rstrip(".").strip().lower()


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
        raise ValueError(f"No numeric value found in '{pred}'")
    numeric_str = match.group(0).replace(",", "")
    result = float(numeric_str)
    # Apply negation if original input started with 'minus' or 'negative'
    if is_negative and not numeric_str.startswith("-"):
        result = -result
    return result


def mean_relative_accuracy(pred, true, start=0.5, end=0.95, step=0.05) -> float:
    thresholds = np.linspace(start, end, int((end - start) / step) + 2)
    rel_err = np.abs(pred - true) / true
    return float(np.mean([(rel_err < 1 - t).mean() for t in thresholds]))


def weighted_mean_std(scores: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """
    Weighted mean:
        weighted_avg = sum(score_i * count_i) / sum(count_i)
    Weighted variance:
        weighted_var = sum(count_i * (score_i - weighted_avg)**2) / sum(count_i)
    Weighted std:
        weighted_std = sqrt(weighted_var)
    Only models with count > 0 are included.
    """
    mask = counts > 0
    scores = scores[mask]
    counts = counts[mask]
    weighted_avg = (scores * counts).sum() / counts.sum() if counts.sum() > 0 else 0
    weighted_var = ((counts * (scores - weighted_avg) ** 2).sum() / counts.sum()) if counts.sum() > 0 else 0
    weighted_std = weighted_var**0.5
    return weighted_avg, weighted_std
