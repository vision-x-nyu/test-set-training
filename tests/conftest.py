"""
Pytest configuration and shared fixtures for TsT tests.
"""

import pytest
from functools import wraps
from TsT.core.benchmark import BenchmarkRegistry


def isolate_registry(func):
    """
    Decorator to isolate benchmark registry state during tests.

    Saves the current registry before the test, clears it for clean testing,
    then restores the original state after the test completes (even if it fails).

    Usage:
        @isolate_registry
        def test_something(self):
            # Registry starts empty, can register test benchmarks
            pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save original registry state
        original_benchmarks = BenchmarkRegistry._benchmarks.copy()
        # Clear registry for clean testing
        BenchmarkRegistry._benchmarks = {}

        try:
            # Run the test
            return func(*args, **kwargs)
        finally:
            # Always restore original state
            BenchmarkRegistry._benchmarks = original_benchmarks

    return wrapper


# Pytest fixtures can also be defined here
@pytest.fixture
def clean_registry():
    """
    Pytest fixture alternative to the decorator.

    Usage:
        def test_something(clean_registry):
            # Registry is automatically cleaned and restored
            pass
    """
    original_benchmarks = BenchmarkRegistry._benchmarks.copy()
    BenchmarkRegistry._benchmarks = {}

    try:
        yield
    finally:
        BenchmarkRegistry._benchmarks = original_benchmarks
