from ._base import DATASET_REGISTRY, _BaseDataset, load_dataset, register_dataset_class
from ._abalone import AbaloneContinuous, AbaloneMixed  # noqa: F401
from ._adult import Adult  # noqa: F401
from ._airfoil import Airfoil  # noqa: F401
from ._algeria import Algeria  # noqa: F401
from ._sachs import (  # noqa: F401
    SachsContinuous,
    SachsContinuousJittered,
    SachsContinuousJitteredLogScale,
    SachsContinuousLogScale,
    SachsDiscrete,
    SachsMixed,
)
from ._boston_housing import BostonHousing  # noqa: F401
from ._wine_quality import (  # noqa: F401
    WineQualityWhite,
    WineQualityRed,
    WineQualityRedWhiteMixed,
)

__all__ = [
    "_BaseDataset",
    "DATASET_REGISTRY",
    "register_dataset_class",
    "load_dataset",
]
