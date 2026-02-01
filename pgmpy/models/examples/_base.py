import gzip
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Union

import requests
from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.base import DAG
from pgmpy.global_vars import PGMPY_DATA_HOME
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFReader


@dataclass
class Model:
    name: str
    model: Union[BayesianNetwork, DAG]
    tags: Dict[str, Any] = None

    def __str__(self) -> str:
        return (
            f"Model(name={self.name}, \n "
            f"model_type={type(self.model)}, \n "
            f"tags={self.tags})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class _BaseModel(BaseObject):
    """
    Base class for all models in pgmpy.
    Inherits from skbase.base.BaseObject to utilize its tag and lookup functionality.
    """

    _tags = {
        "name": None,
        "type": None,
        "file_format": None,
        "n_nodes": None,
        "n_edges": None,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_models/refs/heads/main"

    @classmethod
    def _get_raw_data(cls, data_type: str, url: str) -> bytes:
        """
        Checks if the data is cached locally; if not, fetches it from the URL and caches it.
        """
        name = cls.get_class_tag("name")
        cache_dir_path = os.path.join(
            PGMPY_DATA_HOME,
            hashlib.sha256(f"{name}_{cls.base_url}".encode()).hexdigest(),
        )
        path = os.path.join(cache_dir_path, data_type)
        if os.path.exists(path):
            with open(path, "rb") as f:
                raw_data = f.read()
        else:
            os.makedirs(cache_dir_path, exist_ok=True)
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            raw_data = resp.content
            if raw_data.startswith(b"\x1f\x8b"):
                raw_data = gzip.decompress(raw_data)
            with open(path, "wb") as f:
                f.write(raw_data)
        return raw_data

    @classmethod
    def load_model_object(cls):
        """Fetches/reads from cache the data associated with the model."""
        name = cls.get_class_tag("name")
        file_format = cls.get_class_tag("file_format")
        url = f"{cls.base_url}/{cls.data_url}"
        local_file_name = f"{name}.{file_format}"
        cls._get_raw_data(local_file_name, url)

        cache_dir = os.path.join(
            PGMPY_DATA_HOME,
            hashlib.sha256(f"{name}_{cls.base_url}".encode()).hexdigest(),
        )
        full_path = os.path.join(cache_dir, local_file_name)

        if file_format == "bif":
            return BIFReader(full_path).get_model()
        # elif file_format == "json":
        #     return JSONReader(full_path).get_model()
        elif file_format == "txt":
            with open(full_path, "r") as f:
                return DAG.from_dagitty(f.read())
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


def load_model(name: str):
    """
    Loads an example model by name.

    Parameters
    ----------
    name : str
        Name of the example model to load.
    """
    all_models = all_objects(
        object_types="_BaseModel",
        package_name="pgmpy.models.examples",
        return_names=False,
    )

    target_cls = None
    for cls in all_models:
        if cls.get_class_tag("name") == name:
            target_cls = cls
            break
    if target_cls is None:
        raise ValueError(
            f"Model with name '{name}' not found. Please use list_models() to see available datasets."
        )

    return Model(
        name=target_cls.get_class_tag("name"),
        model=target_cls.load_model_object(),
        tags=target_cls.get_class_tags(),
    )


def list_models(**filter_tags) -> list[str]:
    """
    Lists all available example models.

    Returns
    -------
    list
        List of names of all available example models.
    """
    all_models = all_objects(
        object_types="_BaseModel",
        package_name="pgmpy.models.examples",
        return_names=False,
        filter_tags=filter_tags,
    )

    Model_names = [
        cls.get_class_tag("name")
        for cls in all_models
        if cls.get_class_tag("name") is not None
    ]

    return sorted(Model_names)
