"""
VSI benchmark module.
"""

# Import to trigger registration
from .benchmark import VSIBenchmark

# Export the benchmark
benchmark = VSIBenchmark()

__all__ = ["benchmark"]
