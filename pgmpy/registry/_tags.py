"""
Register of the skbase tags used by pgmpy objects.

Each tag is a subclass of ``_BaseTag``. The class name is the tag name with ``:`` replaced by ``__``, and the class
docstring is the tag's documentation. The ``_tags`` of a tag class describe the tag:

- ``tag_name`` (str): Name of the tag as used in the ``_tags`` of an object.
- ``parent_type`` (list of str): Object types the tag applies to, from ``OBJECT_TYPES``. Every object type's base
  class sets its ``object_type`` tag to one of these.
- ``tag_type``: Expected type of the tag value. One of:

  - ``"bool"``: ``True`` or ``False``.
  - ``"str"``: Any string.
  - ``("str", choices)``: Any element of ``choices``. ``None`` is valid only if it is in ``choices``.
  - ``("list", choices)``: A list whose elements are all in ``choices``. The choices can be classes, e.g. graph
    classes from :mod:`pgmpy.base`.

- ``short_descr`` (str): One-line description of the tag.

New tags only need a new tag class here. ``TAG_REGISTER`` is built from the tag classes.
"""

import inspect
import sys

import pandas as pd
from skbase.base import BaseObject

from pgmpy.base import ADMG, DAG, MAG, PDAG

OBJECT_TYPES = [
    "causal_discovery",
    "ci_test",
    "structure_score",
    "bivariate_score",
    "supervised_metric",
    "unsupervised_metric",
]

METRIC_TYPES = ["supervised_metric", "unsupervised_metric"]

DATA_TYPES = ["discrete", "continuous", "mixed"]


class _BaseTag(BaseObject):
    """Base class for all tags."""

    _tags = {
        "object_type": "tag",
        "tag_name": None,
        "parent_type": None,
        "tag_type": None,
        "short_descr": None,
    }


# ------------------------------------
# Tags shared by several object types
# ------------------------------------


class object_type(_BaseTag):
    """
    Type of the object, e.g. ``"causal_discovery"`` or ``"ci_test"``.

    Set by the base class of each object type. Use it with :func:`pgmpy.registry.all_tags` to list the tags that
    apply to an object.
    """

    _tags = {
        "tag_name": "object_type",
        "parent_type": OBJECT_TYPES,
        "tag_type": ("str", OBJECT_TYPES),
        "short_descr": "Type of the object, e.g. 'causal_discovery' or 'ci_test'.",
    }


class name(_BaseTag):
    """
    Unique lowercase name of the object.

    CI tests, structure scores, bivariate scores, and metrics can be selected by this name, e.g.
    ``PC(ci_test="pearsonr")``, ``GES(scoring_method="bic-g")``, or
    ``PC(return_type="dag").fit(data).score(X=data, metric="correlation_score")``.
    """

    _tags = {
        "tag_name": "name",
        "parent_type": OBJECT_TYPES,
        "tag_type": "str",
        "short_descr": "Unique lowercase name of the object.",
    }


class data_types(_BaseTag):
    """
    Data types the object can be applied to.

    For causal discovery algorithms that use a CI test, a structure score, a bivariate score, or a pairwise estimator,
    the class-level value is every data type supported by some available component. ``fit`` narrows it to the data
    types that every component used for fitting also supports.

    The value is a list rather than a tuple because ``skbase.lookup.all_objects`` only matches single elements of a
    tag value in ``filter_tags`` when the value is a list.
    """

    _tags = {
        "tag_name": "data_types",
        "parent_type": ["causal_discovery", "ci_test", "structure_score", "bivariate_score"],
        "tag_type": ("list", DATA_TYPES),
        "short_descr": "Data types the object can be applied to.",
    }


class default_for(_BaseTag):
    """
    What the object is used by default for, or ``None``.

    - CI tests and structure scores: the data type for which the object is selected when a causal discovery algorithm
      is given ``ci_test=None`` or ``scoring_method=None``.
    - Metrics: ``"supervised"`` or ``"unsupervised"``, for the metric used by
      :meth:`pgmpy.causal_discovery._base.BaseCausalDiscovery.score` when ``metric=None``.
    """

    _tags = {
        "tag_name": "default_for",
        "parent_type": ["ci_test", "structure_score"] + METRIC_TYPES,
        "tag_type": ("str", DATA_TYPES + ["supervised", "unsupervised", None]),
        "short_descr": "What the object is used by default for, or None.",
    }


class requires_data(_BaseTag):
    """
    Whether the object needs data.

    ``False`` for oracle CI tests such as ``IndependenceMatch`` and for metrics that only compare two graphs.
    """

    _tags = {
        "tag_name": "requires_data",
        "parent_type": ["ci_test"] + METRIC_TYPES,
        "tag_type": "bool",
        "short_descr": "Whether the object needs data.",
    }


class is_symmetric(_BaseTag):
    """
    Whether swapping the two inputs leaves the result unchanged.

    For CI tests, swapping ``X`` and ``Y``. For supervised metrics, swapping the true and the estimated graph.
    """

    _tags = {
        "tag_name": "is_symmetric",
        "parent_type": ["ci_test", "supervised_metric"],
        "tag_type": "bool",
        "short_descr": "Whether swapping the two inputs leaves the result unchanged.",
    }


# ---------------------------
# Causal discovery algorithms
# ---------------------------


class identifiable_graph(_BaseTag):
    """
    How far the causal discovery algorithm can identify the causal graph.

    - ``"dag"``: Every edge is oriented, e.g. by relying on functional or noise assumptions (``ANM``), on patterns in
      the data (``VarSort``), or on an expert (``ExpertInLoop``).
    - ``"cpdag"``: The graph is identified only up to its Markov equivalence class, e.g. ``PC`` and ``GES``. With
      expert knowledge, more edges can be oriented (a maximally oriented PDAG).
    - ``"pag"``: The graph is identified only up to a partial ancestral graph, for methods that allow latent
      variables.

    This describes the method, not the type of ``causal_graph_``: an algorithm with a ``return_type`` parameter can
    convert its result, e.g. ``PC(return_type="dag")`` returns one arbitrary DAG from the equivalence class.
    """

    _tags = {
        "tag_name": "identifiable_graph",
        "parent_type": ["causal_discovery"],
        "tag_type": ("str", ["dag", "cpdag", "pag"]),
        "short_descr": "How far the algorithm can identify the causal graph: 'dag', 'cpdag', or 'pag'.",
    }


class requires_target(_BaseTag):
    """Whether the causal discovery algorithm requires a target variable to be specified, e.g. ``TAN``."""

    _tags = {
        "tag_name": "requires_target",
        "parent_type": ["causal_discovery"],
        "tag_type": "bool",
        "short_descr": "Whether the algorithm requires a target variable to be specified.",
    }


class capability__multivariate(_BaseTag):
    """
    Whether the causal discovery algorithm can learn a graph over more than two variables.

    ``False`` for bivariate methods, such as ``ANM`` and ``IGCI``, that orient the edge between exactly two
    variables.
    """

    _tags = {
        "tag_name": "capability:multivariate",
        "parent_type": ["causal_discovery"],
        "tag_type": "bool",
        "short_descr": "Whether the algorithm can learn a graph over more than two variables.",
    }


class capability__expert_knowledge(_BaseTag):
    """Constraints from :class:`pgmpy.causal_discovery.ExpertKnowledge` that the causal discovery algorithm respects."""

    _tags = {
        "tag_name": "capability:expert_knowledge",
        "parent_type": ["causal_discovery"],
        "tag_type": ("list", ["forbidden_edges", "required_edges", "search_space", "temporal_order"]),
        "short_descr": "Constraints from ExpertKnowledge that the algorithm respects.",
    }


# -----------
# Assumptions
# -----------
#
# Each assumption tag is True if the method's guarantees require the assumption and False otherwise. For causal
# discovery algorithms that use a CI test, a structure score, a bivariate score, or a pairwise estimator, the
# class-level value is the union over all available components (False if some component doesn't require it). `fit`
# sets it to True if the algorithm or any component used for fitting requires it. Assumptions that can't be expressed
# this way (for example, an assumption that holds if either of two conditions does) are described in the method's
# docstring.


class assumption__causal_sufficiency(_BaseTag):
    """Whether the method assumes there are no unobserved common causes of two or more observed variables."""

    _tags = {
        "tag_name": "assumption:causal_sufficiency",
        "parent_type": ["causal_discovery"],
        "tag_type": "bool",
        "short_descr": "Assumes no unobserved common causes of observed variables.",
    }


class assumption__acyclicity(_BaseTag):
    """Whether the method assumes the true causal graph has no directed cycles."""

    _tags = {
        "tag_name": "assumption:acyclicity",
        "parent_type": ["causal_discovery"],
        "tag_type": "bool",
        "short_descr": "Assumes the true causal graph has no directed cycles.",
    }


class assumption__faithfulness(_BaseTag):
    """
    Whether the method assumes every conditional independence in the data is implied by the true causal graph.

    Methods that need a weaker assumption, such as the sparsest Markov representation assumption of ``SP``, set this
    to ``False`` and describe the weaker assumption in their docstring.
    """

    _tags = {
        "tag_name": "assumption:faithfulness",
        "parent_type": ["causal_discovery"],
        "tag_type": "bool",
        "short_descr": "Assumes every conditional independence is implied by the true graph.",
    }


class assumption__linearity(_BaseTag):
    """Whether the method assumes each variable is a linear function of its parents."""

    _tags = {
        "tag_name": "assumption:linearity",
        "parent_type": ["causal_discovery", "ci_test", "structure_score", "bivariate_score"],
        "tag_type": "bool",
        "short_descr": "Assumes each variable is a linear function of its parents.",
    }


class assumption__additive_noise(_BaseTag):
    """Whether the method assumes each variable is a function of its parents plus independent, additive noise."""

    _tags = {
        "tag_name": "assumption:additive_noise",
        "parent_type": ["causal_discovery", "ci_test", "structure_score", "bivariate_score"],
        "tag_type": "bool",
        "short_descr": "Assumes independent, additive noise.",
    }


class assumption__gaussian_noise(_BaseTag):
    """Whether the method assumes the noise terms are Gaussian."""

    _tags = {
        "tag_name": "assumption:gaussian_noise",
        "parent_type": ["causal_discovery", "ci_test", "structure_score", "bivariate_score"],
        "tag_type": "bool",
        "short_descr": "Assumes Gaussian noise.",
    }


class assumption__non_gaussian_noise(_BaseTag):
    """Whether the method assumes the noise terms are non-Gaussian, e.g. LiNGAM."""

    _tags = {
        "tag_name": "assumption:non_gaussian_noise",
        "parent_type": ["causal_discovery"],
        "tag_type": "bool",
        "short_descr": "Assumes non-Gaussian noise.",
    }


class assumption__low_noise(_BaseTag):
    """Whether the method assumes the effect is a (nearly) deterministic function of its cause, e.g. ``IGCI``."""

    _tags = {
        "tag_name": "assumption:low_noise",
        "parent_type": ["causal_discovery"],
        "tag_type": "bool",
        "short_descr": "Assumes the effect is a (nearly) deterministic function of its cause.",
    }


# ----------------
# Bivariate scores
# ----------------


class input_type(_BaseTag):
    """
    What the two inputs of the bivariate score are.

    - ``"cause_residual"``: A candidate cause and the residuals of regressing the effect on it, as passed by ``ANM``.
      The score measures how dependent the residuals are on the cause.
    - ``"cause_effect"``: A candidate cause and effect, as passed by ``IGCI``. The score must be asymmetric, since the
      algorithm compares ``score(x, y)`` with ``score(y, x)``.

    Algorithms only accept built-in scores of the input type they pass.
    """

    _tags = {
        "tag_name": "input_type",
        "parent_type": ["bivariate_score"],
        "tag_type": ("str", ["cause_residual", "cause_effect"]),
        "short_descr": "What the two inputs of the bivariate score are.",
    }


# -------
# Metrics
# -------


class requires_true_graph(_BaseTag):
    """Whether the metric compares an estimated graph against a true graph (supervised) rather than against data."""

    _tags = {
        "tag_name": "requires_true_graph",
        "parent_type": METRIC_TYPES,
        "tag_type": "bool",
        "short_descr": "Whether the metric needs a true graph.",
    }


class supported_graph_types(_BaseTag):
    """Graph classes from :mod:`pgmpy.base` that the metric accepts."""

    _tags = {
        "tag_name": "supported_graph_types",
        "parent_type": METRIC_TYPES,
        "tag_type": ("list", [DAG, PDAG, MAG, ADMG]),
        "short_descr": "Graph classes the metric accepts.",
    }


class output_type(_BaseTag):
    """
    Type of the value returned by ``evaluate``.

    - ``"scalar"``: A single number.
    - ``"tuple"``: A tuple of numbers, e.g. ``FisherC(compute_rmsea=True)`` returns the p-value and the RMSEA.
    - ``"dict"``: A dict of named results, e.g. precision and recall from ``AdjacencyConfusionMatrix``.
    - ``"dataframe"``: A ``pandas.DataFrame``, e.g. one row per implied CI from ``ImpliedCIs``.

    If the output depends on a hyperparameter, the class-level value is the default output and ``__init__`` sets it for
    the instance.
    """

    _tags = {
        "tag_name": "output_type",
        "parent_type": METRIC_TYPES,
        "tag_type": ("str", ["scalar", "tuple", "dict", "dataframe"]),
        "short_descr": "Type of the value returned by evaluate.",
    }


class lower_is_better(_BaseTag):
    """
    Whether a lower value of the metric means a better graph.

    Only meaningful when ``output_type`` is ``"scalar"``; ``False`` otherwise.
    """

    _tags = {
        "tag_name": "lower_is_better",
        "parent_type": METRIC_TYPES,
        "tag_type": "bool",
        "short_descr": "Whether a lower value means a better graph (scalar output only).",
    }


def _build_tag_register() -> list[tuple]:
    register = []
    # vars() keeps the definition order, so tags are listed in the order they are defined in this module.
    for cls in list(vars(sys.modules[__name__]).values()):
        if inspect.isclass(cls) and issubclass(cls, _BaseTag) and cls is not _BaseTag:
            tags = cls.get_class_tags()
            for parent_type in tags["parent_type"]:
                register.append((tags["tag_name"], parent_type, tags["tag_type"], tags["short_descr"]))
    return register


# One (tag name, object type, tag type, short description) tuple for each (tag, object type) pair.
TAG_REGISTER = _build_tag_register()


def all_tags(object_types: str | list[str] | None = None, as_dataframe: bool = False) -> list[tuple] | pd.DataFrame:
    """
    List the tags that apply to the given object types.

    Parameters
    ----------
    object_types : str or list of str, optional
        Object types from ``OBJECT_TYPES``, e.g. ``"causal_discovery"``. If ``None``, the tags of all object types are
        returned.

    as_dataframe : bool, default=False
        If ``True``, return a ``pandas.DataFrame`` with columns ``name``, ``object_type``, ``type``, and
        ``description``. Otherwise, return a list of tuples in the same order.

    Returns
    -------
    list of tuple or pandas.DataFrame
        One entry for each (tag, object type) pair.

    Examples
    --------
    >>> from pgmpy.registry import all_tags
    >>> tags = all_tags("ci_test", as_dataframe=True)
    >>> "requires_data" in tags["name"].values
    True
    """
    if isinstance(object_types, str):
        object_types = [object_types]
    tags = [tag for tag in TAG_REGISTER if object_types is None or tag[1] in object_types]
    if as_dataframe:
        return pd.DataFrame(tags, columns=["name", "object_type", "type", "description"])
    return tags


def check_tag_is_valid(tag_name: str, tag_value) -> None:
    """
    Check that a tag value has the type registered for the tag.

    Parameters
    ----------
    tag_name : str
        Name of the tag.

    tag_value : object
        Value of the tag.

    Raises
    ------
    KeyError
        If ``tag_name`` is not a registered tag.

    ValueError
        If ``tag_value`` doesn't have the registered type of the tag.

    Examples
    --------
    >>> from pgmpy.registry import check_tag_is_valid
    >>> check_tag_is_valid("data_types", ["discrete", "continuous"])
    >>> check_tag_is_valid("requires_target", "yes")
    Traceback (most recent call last):
    ...
    ValueError: Tag 'requires_target' must be a bool, got 'yes'.
    """
    tag_types = {tag[0]: tag[2] for tag in TAG_REGISTER}
    if tag_name not in tag_types:
        raise KeyError(f"'{tag_name}' is not a registered tag.")
    tag_type = tag_types[tag_name]

    if tag_type == "bool":
        valid, expected = isinstance(tag_value, bool), "a bool"
    elif tag_type == "str":
        valid, expected = isinstance(tag_value, str), "a str"
    elif tag_type[0] == "str":
        valid, expected = tag_value in tag_type[1], f"one of {tag_type[1]}"
    else:
        valid = isinstance(tag_value, list) and set(tag_value) <= set(tag_type[1])
        expected = f"a list with elements from {tag_type[1]}"

    if not valid:
        raise ValueError(f"Tag '{tag_name}' must be {expected}, got {tag_value!r}.")
