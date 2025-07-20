from .models import (
    Count2DModel,
    Relation2DModel,
    Depth3DModel,
    Distance3DModel,
)
from .data_loader import load_data


def get_models():
    """Get all CVB benchmark models."""
    return [
        # MC
        Count2DModel(),
        Relation2DModel(),
        Depth3DModel(),
        Distance3DModel(),
    ]


__all__ = ["load_data", "get_models"]
