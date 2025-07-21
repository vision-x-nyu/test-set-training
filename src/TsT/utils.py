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
    converted_pred = t2d.convert(cleaned_pred)
    # Remove all non-numeric, non-dot, and non-minus characters (e.g., '$99.99', '47.1%', '99 cents')
    numeric_str = re.sub(r"[^0-9.\-]", "", converted_pred)
    result = float(numeric_str)
    # Negate if original input started with 'minus' or 'negative'
    return -result if is_negative else result
