"""
MMMU benchmark module.
"""

# Import to trigger registration
from .benchmark import MMMUBenchmark

# Export the benchmark
benchmark = MMMUBenchmark()

__all__ = ["benchmark"]
