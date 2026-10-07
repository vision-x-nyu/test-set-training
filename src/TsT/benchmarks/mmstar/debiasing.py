"""IBP strategy for the MMStar benchmark.

Preserves diversity across the 6 high-level categories (``category`` column)
during iterative bias pruning. No result in the paper uses IBP on MMStar.
"""

from TsT.debiasing.strategies import TopKDiversityStrategy


class MMStarIBPStrategy(TopKDiversityStrategy):
    """IBP strategy for MMStar — groups by high-level ``category`` (min 5)."""

    def __init__(self, min_samples_per_category: int = 5, tie_break: str = "id") -> None:
        super().__init__(
            group_column="category",
            min_samples_per_group=min_samples_per_category,
            id_col="index",
            tie_break=tie_break,
        )
