"""
CVB benchmark module.
"""

# Import to trigger registration
from .benchmark import CVBBenchmark

# Export the benchmark
benchmark = CVBBenchmark()

__all__ = ["benchmark"]
