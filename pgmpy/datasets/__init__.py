from ._base import DATASETS, DatasetRegistry, _BaseDataset, dataset_class, load_dataset
from .abalone import Abalone  # noqa: F401
from .sachs import Sachs  # noqa: F401

__all__ = [
    "_BaseDataset",
    "DATASETS",
    "DatasetRegistry",
    "dataset_class",
    "load_dataset",
]
