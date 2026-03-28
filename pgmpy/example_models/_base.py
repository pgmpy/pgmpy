import gzip
import io

from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.base import DAG
from pgmpy.readwrite import BIFReader
from pgmpy.utils.hf_hub import read_hf_file


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

    repo_id = "pgmpy/example_models"
    revision = "main"

    @classmethod
    def _get_raw_data(cls) -> bytes:
        """
        Fetches the model file from the Hugging Face Hub cache.
        """
        return read_hf_file(
            repo_id=cls.repo_id,
            filename=cls.data_url,
            revision=cls.revision,
        )


class DiscreteMixin:
    """
    Mixin class for loading discrete Bayesian networks from BIF files.
    """

    @classmethod
    def load_model_object(cls):
        return BIFReader(string=gzip.decompress(cls._get_raw_data()).decode("utf-8")).get_model()


class BIFMixin:
    """
    Mixin class for loading discrete Bayesian networks from plain (non-gzipped) BIF files.
    """

    @classmethod
    def load_model_object(cls):
        return BIFReader(string=cls._get_raw_data().decode("utf-8")).get_model()


class ContinuousMixin:
    """
    Mixin class for loading continuous Bayesian networks from JSON files.
    """

    @classmethod
    def load_model_object(cls):
        from pgmpy.models import LinearGaussianBayesianNetwork

        raw_data = cls._get_raw_data()
        file_obj = io.BytesIO(raw_data)
        return LinearGaussianBayesianNetwork.load(file_obj)


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
    model: pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork or
           pgmpy.models.LinearGaussianBayesianNetwork or
           pgmpy.models.FunctionalBayesianNetwork
        The loaded example model.

    Examples
    --------
    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/alarm")
    >>> print(model)
    DiscreteBayesianNetwork named 'unknown' with 37 nodes and 46 edges
    >>> len(model.nodes())
    37
    >>> model.get_cpds("HISTORY")
    <TabularCPD representing P(HISTORY:2 | LVFAILURE:2) at 0x7d4527a84230>

    >>> model = load_model("dagitty/acid_1996")
    >>> print(model)
    DAG with 18 nodes and 22 edges

    >>> model = load_model("bnlearn/arth150")
    >>> print(model)
    LinearGaussianBayesianNetwork with 107 nodes and 150 edges

    >>> model = load_model("bnrep/asia")
    >>> print(model)
    DiscreteBayesianNetwork named 'unknown' with 8 nodes and 8 edges
    """
    target_model = all_objects(
        object_types=_BaseExampleModel,
        package_name="pgmpy.example_models",
        filter_tags={"name": name},
        return_names=False,
    )

    if not target_model:
        raise ValueError(f"Model with name '{name}' not found. Please use list_models() to see available datasets.")

    return target_model[0].load_model_object()


def list_models(**filter_tags) -> list[str]:
    """
    Lists all available example models.

    The models can be filtered based on their tags by providing keyword
    arguments. Supports both exact matching and comparator-based suffixes
    for numeric tags (``n_nodes``, ``n_edges``).

    Available tags
    --------------
    - name            : str  -- exact model name
    - n_nodes         : int  -- number of nodes
    - n_edges         : int  -- number of edges
    - is_parameterized: bool -- has CPDs / parameters defined
    - is_discrete     : bool -- discrete variables only
    - is_continuous   : bool -- continuous variables only
    - is_hybrid       : bool -- both discrete and continuous variables

    Comparator suffixes (for numeric tags)
    ---------------------------------------
    ``__gt``   strictly greater than     ``n_nodes__gt=10``
    ``__gte``  greater than or equal to  ``n_nodes__gte=10``
    ``__lt``   strictly less than        ``n_nodes__lt=50``
    ``__lte``  less than or equal to     ``n_nodes__lte=50``
    ``__ne``   not equal to              ``n_nodes__ne=10``
    ``__in``   value in collection       ``n_nodes__in=[10, 20, 46]``

    Filters are combined with logical AND. Suffixed and plain filters may
    be freely mixed: ``list_models(n_nodes__gte=10, is_discrete=True)``.

    Parameters
    ----------
    **filter_tags
        Tag-based filter criteria (see above).

    Returns
    -------
    list of str
        Sorted list of model names matching all supplied criteria.

    Examples
    --------
    >>> from pgmpy.example_models import list_models
    >>> list_models()
    ['bnlearn/alarm', 'bnlearn/arth150', ...]
    >>> list_models(is_discrete=True)
    ['bnlearn/alarm', 'bnlearn/asia', ...]
    >>> list_models(n_nodes=10)
    [...]
    >>> list_models(n_nodes__gt=10)
    [...]
    >>> list_models(n_nodes__gte=10, n_nodes__lte=50)
    [...]
    >>> list_models(n_nodes__in=[10, 20, 46], is_discrete=True)
    [...]
    """
    from pgmpy.utils.filter_utils import apply_comparator_filters, split_filter_tags

    valid_tags = set(_BaseExampleModel._tags.keys())

    exact_tags, comparator_filters = split_filter_tags(filter_tags, valid_tags)

    all_models = all_objects(
        object_types=_BaseExampleModel,
        package_name="pgmpy.example_models",
        return_names=False,
        filter_tags=exact_tags,
    )

    all_models = apply_comparator_filters(all_models, comparator_filters)

    model_names = [cls.get_class_tag("name") for cls in all_models if cls.get_class_tag("name") is not None]

    return sorted(model_names)
