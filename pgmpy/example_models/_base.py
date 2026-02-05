import gzip
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Union

from skbase.base import BaseObject
from skbase.lookup import all_objects
from skbase.utils.dependencies import _safe_import

from pgmpy.base import DAG
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.global_vars import PGMPY_DATA_HOME
from pgmpy.models import BayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.readwrite import BIFReader

requests = _safe_import("requests")


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


class _BaseExampleModel(BaseObject):
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
            with open(path, "wb") as f:
                f.write(raw_data)
        return raw_data

    @staticmethod
    def clear_cache():
        """
        Clears the cached data for all models.
        """
        if os.path.exists(PGMPY_DATA_HOME):
            shutil.rmtree(PGMPY_DATA_HOME)


class DiscreteExampleMixin:
    @classmethod
    def load_model_object(cls):
        """Fetches/reads from cache the data associated with the discrete model."""
        name = cls.get_class_tag("name")
        file_format = cls.get_class_tag("file_format")
        url = f"{cls.base_url}/{cls.data_url}"
        compressed_file_name = f"{name}.{file_format}.gz"
        cls._get_raw_data(compressed_file_name, url)

        cache_dir = os.path.join(
            PGMPY_DATA_HOME,
            hashlib.sha256(f"{name}_{cls.base_url}".encode()).hexdigest(),
        )
        compressed_path = os.path.join(cache_dir, compressed_file_name)
        if file_format == "bif":
            with gzip.open(compressed_path, "rt", encoding="utf-8") as f:
                bif_text = f.read()
            reader = BIFReader(string=bif_text)
            return reader.get_model()

        else:
            raise ValueError(f"Unsupported file format: {file_format}")


class ContinuousExampleMixin:

    @classmethod
    def load_model_object(cls):
        """Fetches/reads from cache the data associated with the continuous model."""
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
        if file_format == "json":
            with open(full_path, "r") as f:
                data = json.load(f)
            # Extract nodes, arcs, and CPDs from the JSON file
            nodes = data.get("nodes")
            arcs = data.get("arcs")
            cpds_data = data.get("cpds")

            model = LinearGaussianBayesianNetwork(arcs)
            model.add_nodes_from(nodes)

            cpds = []
            for node, cpd_info in cpds_data.items():
                coefficients = cpd_info["coefficients"]
                var = cpd_info["variance"][0]
                parents = cpd_info["parents"]

                intercept = coefficients["(Intercept)"][0]

                parent_coeffs = [coefficients[parent][0] for parent in parents]

                cpd = LinearGaussianCPD(
                    variable=node,
                    beta=[intercept] + parent_coeffs,
                    std=math.sqrt(var),
                    evidence=parents,
                )
                cpds.append(cpd)

            model.add_cpds(*cpds)
            return model
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


class DAGExampleMixin:
    @classmethod
    def load_model_object(cls):
        """Fetches/reads from cache the data associated with the DAG model."""
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
        if file_format == "txt":
            with open(full_path, "r") as f:
                return DAG.from_dagitty(string=f.read())
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
        object_types=_BaseExampleModel,
        package_name="pgmpy.models.example_models",
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
        object_types=_BaseExampleModel,
        package_name="pgmpy.models.example_models",
        return_names=False,
        filter_tags=filter_tags,
    )

    Model_names = [
        cls.get_class_tag("name")
        for cls in all_models
        if cls.get_class_tag("name") is not None
    ]

    return sorted(Model_names)
