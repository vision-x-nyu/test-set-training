"""
IBP strategy for MMMU benchmark.

Preserves diversity across academic subfields (``subfield_truncated`` column)
during iterative bias pruning.
"""

from TsT.debiasing.strategies import TopKDiversityStrategy


class MMMUIBPStrategy(TopKDiversityStrategy):
    """IBP strategy for MMMU.

    Groups by ``subfield_truncated`` — subfields with fewer than 5 samples
    in the original dataset are already grouped as "Other" by the benchmark
    loader, so the diversity constraint operates on those merged groups.

    Args:
        min_samples_per_subfield: Minimum samples to preserve per subfield.
            Default 5 matches the benchmark's own grouping threshold.
    """

    def __init__(self, min_samples_per_subfield: int = 5, tie_break: str = "id") -> None:
        super().__init__(
            group_column="subfield_truncated",
            min_samples_per_group=min_samples_per_subfield,
            id_col="id",
            tie_break=tie_break,
        )
