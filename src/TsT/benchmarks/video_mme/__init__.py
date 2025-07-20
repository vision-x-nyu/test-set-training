from .models import VideoMMEModel
from .data_loader import load_data


def get_models():
    """Get all Video-MME benchmark models."""
    return [
        # MC
        VideoMMEModel(),
    ]


__all__ = ["load_data", "get_models"]
