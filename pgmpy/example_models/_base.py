import gzip
import hashlib
import json
import math
import os
import shutil
from urllib.request import urlopen

from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.base import DAG
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.global_vars import PGMPY_DATA_HOME
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.readwrite import BIFReader


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

            with urlopen(f"{cls.base_url}/{cls.data_url}", timeout=60) as response:
                raw_data = response.read()

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
    """
    Mixin class for loading discrete Bayesian networks from BIF files.
    """

    @classmethod
    def load_model_object(cls):
        return BIFReader(
            string=gzip.decompress(cls._get_raw_data()).decode("utf-8")
        ).get_model()


class ContinuousMixin:
    """
    Mixin class for loading continuous Bayesian networks from JSON files.
    """

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
    """
    Mixin class for loading DAGs from dagitty string format.
    """

    @classmethod
    def load_model_object(cls):
        return DAG.from_dagitty(string=cls._get_raw_data().decode("utf-8"))


def load_model(name: str):
    """
    Loads an example model by name.

    To find all available example models, use the `list_models()` function.

    Parameters
    ----------
    name : str
        Name of the example model to load.

    Returns
    -------
    model: pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork or pgmpy.models.LinearGaussianBayesianNetwork or
                pgmpy.models.FunctionalBayesianNetwork
        The loaded example model.

    Examples
    --------
    #  Loading a discrete Bayesian network with parameters.

    >>> from pgmpy.example_models import load_model
    >>> model = load_model("alarm")
    >>> print(model)
    DiscreteBayesianNetwork named 'unknown' with 37 nodes and 46 edges
    >>> len(model.nodes())
    37
    >>> model.get_cpds("HISTORY")
    <TabularCPD representing P(HISTORY:2 | LVFAILURE:2) at 0x7d4527a84230>

    # Loading a DAG without parameters.

    >>> model = load_model("acid_1996")
    >>> print(model)
    DAG with 18 nodes and 22 edges
    >>> len(model.nodes())
    18

    # Loading a continuous Bayesian network with parameters.

    >>> model = load_model("arht150")
    >>> print(model)
    LinearGaussianBayesianNetwork with 107 nodes and 150 edges
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


    The models can be filtered based on their tags by providing keyword arguments. The available tags are:
    - name: str
    - n_nodes: No. of nodes in the model.
    - n_edges: No. of edges in the model.
    - is_parameterized: Whether it is just the network structure or also has parameters (CPDs) defined.
    - is_discrete: Whether the model has only discrete variables / parameterization.
    - is_continuous: Whether the model has only continuous variables / parameterization.
    - is_hybrid: Whether the model has both discrete and continuous variables / parameterization.

    Returns
    -------
    list
        List of names of all available example models.

    Examples
    --------
    >>> from pgmpy.example_models import list_models
    >>> list_models()
    ['alarm', 'arth150', ..... ]
    >>> list_models(is_discrete=True)
    ['alarm', 'asia', 'cancer', ..... ]
    >>> list_models(is_parameterized=False)
    ['acid_1996', ...., ]
    """
    all_models = all_objects(
        object_types=_BaseExampleModel,
        package_name="pgmpy.example_models",
        return_names=False,
        filter_tags=filter_tags,
    )

    model_names = [
        cls.get_class_tag("name")
        for cls in all_models
        if cls.get_class_tag("name") is not None
    ]

    return sorted(model_names)
