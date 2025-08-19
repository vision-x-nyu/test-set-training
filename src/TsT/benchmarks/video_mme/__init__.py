"""
Video-MME benchmark module.
"""

# Import to trigger registration
from .benchmark import VideoMMEBenchmark

# Export the benchmark
benchmark = VideoMMEBenchmark()

__all__ = ["benchmark"]
