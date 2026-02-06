import gzip
import hashlib
import json
import math
import os
import shutil

from skbase.base import BaseObject
from skbase.lookup import all_objects
from skbase.utils.dependencies import _safe_import

from pgmpy.base import DAG
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.global_vars import PGMPY_DATA_HOME
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.readwrite import BIFReader

requests = _safe_import("requests")


class _BaseExampleModel(BaseObject):
    """
    Base class for all models in pgmpy.

    Inherits from `skbase.base.BaseObject` to utilize its tag and lookup functionality.
    """

    _tags = {
        "name": bool,
        "n_nodes": None,
        "n_edges": None,
        "is_parameterized": bool,
        "is_discrete": bool,
        "is_continuous": bool,
        "is_hybrid": bool,
    }

    base_url = "https://raw.githubusercontent.com/pgmpy/example_models/refs/heads/main"

    @classmethod
    def _get_raw_data(cls) -> bytes:
        """
        Checks if the data is cached locally; if not, fetches it from the URL and caches it.
        """
        name = cls.get_class_tag("name")
        path = os.path.join(
            PGMPY_DATA_HOME,
            hashlib.sha256(f"{cls.base_url}_{name}".encode()).hexdigest(),
        )
        file_path = os.path.join(path, "model")

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                raw_data = f.read()
        else:
            os.makedirs(path, exist_ok=True)
            resp = requests.get(f"{cls.base_url}/{cls.data_url}", timeout=60)
            resp.raise_for_status()
            raw_data = resp.content
            with open(file_path, "wb") as f:
                f.write(raw_data)
        return raw_data

    @staticmethod
    def clear_cache():
        """
        Clears the cached data for all models.
        """
        if os.path.exists(PGMPY_DATA_HOME):
            shutil.rmtree(PGMPY_DATA_HOME)


class DiscreteMixin:
    @classmethod
    def load_model_object(cls):
        return BIFReader(
            string=gzip.decompress(cls._get_raw_data()).decode("utf-8")
        ).get_model()


class ContinuousMixin:
    @classmethod
    def load_model_object(cls):
        data = json.loads(cls._get_raw_data().decode("utf-8"))
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


class DAGMixin:
    @classmethod
    def load_model_object(cls):
        return DAG.from_dagitty(string=cls._get_raw_data().decode("utf-8"))


def load_model(name: str):
    """
    Loads an example model by name.

    Parameters
    ----------
    name : str
        Name of the example model to load.
    """
    target_model = all_objects(
        object_types=_BaseExampleModel,
        package_name="pgmpy.example_models",
        filter_tags={"name": name},
        return_names=False,
    )

    if target_model is None:
        raise ValueError(
            f"Model with name '{name}' not found. Please use list_models() to see available datasets."
        )

    return target_model[0].load_model_object()


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

    model_names = [
        cls.get_class_tag("name")
        for cls in all_models
        if cls.get_class_tag("name") is not None
    ]

    return sorted(model_names)
