"""
IBP strategy for Video-MME benchmark.

Preserves diversity across task types (``question_type`` column, derived
from ``task_type``) during iterative bias pruning.
"""

from TsT.debiasing.strategies import TopKDiversityStrategy


class VideoMMEIBPStrategy(TopKDiversityStrategy):
    """IBP strategy for Video-MME.

    Groups by ``question_type`` — Video-MME has 10+ task types with
    variable sample counts. The conservative default threshold reflects
    the dataset's diversity.

    Args:
        min_samples_per_type: Minimum samples to preserve per task type.
            Default 5 is conservative given the many types with variable
            sample counts.
    """

    def __init__(self, min_samples_per_type: int = 5, tie_break: str = "id") -> None:
        super().__init__(
            group_column="question_type",
            min_samples_per_group=min_samples_per_type,
            id_col="question_id",
            tie_break=tie_break,
        )
