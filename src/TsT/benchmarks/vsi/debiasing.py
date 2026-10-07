"""
IBP strategy for VSI-Bench.

Preserves diversity across question types (``question_type`` column)
during iterative bias pruning.

Note: this is the automated, TsT-based IBP strategy. The released VSI-Bench-Debiased v1 was
built differently, by hand-written per-type filters (``legacy/debias_vsi_clean.py``).
"""

from TsT.debiasing.strategies import TopKDiversityStrategy


class VSIIBPStrategy(TopKDiversityStrategy):
    """TsT-based IBP strategy for VSI-Bench.

    Groups by ``question_type``: the 10 dataset question types (relative direction is split
    into easy, medium and hard), across numerical and multiple-choice formats. Ties in the
    bias score are broken by question ``id`` (``tie_break="legacy"`` reproduces the
    proceedings' unstable sort).

    Args:
        min_samples_per_type: Minimum samples to preserve per question type.
            Default 10 keeps every question type represented.
    """

    def __init__(self, min_samples_per_type: int = 10, tie_break: str = "id") -> None:
        super().__init__(
            group_column="question_type",
            min_samples_per_group=min_samples_per_type,
            id_col="id",
            tie_break=tie_break,
        )
