from typing import Protocol, List, Literal
import pandas as pd

from .core.protocols import BiasModel


class QType(BiasModel, Protocol):
    """
    Protocol for feature-based bias detection models (e.g., Random Forest).

    Extends BiasModel with feature engineering capabilities required for
    traditional ML approaches.
    """

    feature_cols: List[str]

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        """Collect any statistics derived from *train* only (for leakage‑free CV)."""
        ...

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a **copy** of `df` with question‑specific feature columns added."""
        ...

    @property
    def task(self) -> Literal["clf", "reg"]:
        if self.format == "mc":
            return "clf"
        elif self.format == "num":
            return "reg"
        else:
            raise ValueError(f"Unknown format: {self.format}")

    @property
    def metric(self) -> Literal["acc", "mra"]:
        if self.format == "mc":
            return "acc"
        elif self.format == "num":
            return "mra"
        else:
            raise ValueError(f"Unknown format: {self.format}")
