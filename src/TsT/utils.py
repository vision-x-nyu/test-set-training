from functools import lru_cache
import re

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
