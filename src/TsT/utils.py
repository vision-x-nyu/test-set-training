from functools import lru_cache

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
    converted_pred = t2d.convert(cleaned_pred)
    # Remove spaces and commas to allow float conversion (e.g., '2,000' -> '2000')
    return float(converted_pred.strip().replace(" ", "").replace(",", ""))
