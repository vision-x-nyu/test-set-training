from .models import (
    ObjCountModel,
    ObjAbsDistModel,
    ObjSizeEstModel,
    RoomSizeEstModel,
    RelDistanceModel,
    RelDirModel,
    RoutePlanningModel,
    ObjOrderModel,
)
from .data_loader import load_data

def get_models():
    """Get all VSI benchmark models."""
    return [
        # NUM
        ObjCountModel(),
        ObjAbsDistModel(),
        ObjSizeEstModel(),
        RoomSizeEstModel(),
        # MC
        RelDistanceModel(),
        RelDirModel(),
        RoutePlanningModel(),
        ObjOrderModel(),
    ] 