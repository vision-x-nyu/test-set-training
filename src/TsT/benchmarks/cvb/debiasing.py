"""
IBP strategy for CV-Bench.

Preserves diversity across question types (``question_type`` column)
during iterative bias pruning.
"""

from TsT.debiasing.strategies import TopKDiversityStrategy


class CVBIBPStrategy(TopKDiversityStrategy):
    """IBP strategy for CV-Bench.

    Groups by ``question_type`` — CV-Bench has 4 types (count_2d,
    relation_2d, depth_3d, distance_3d), all multiple-choice.

    Args:
        min_samples_per_type: Minimum samples to preserve per question type.
            Default 10 ensures adequate representation across all 4 types.
    """

    def __init__(self, min_samples_per_type: int = 10, tie_break: str = "id") -> None:
        super().__init__(
            group_column="question_type",
            min_samples_per_group=min_samples_per_type,
            id_col="idx",
            tie_break=tie_break,
        )
