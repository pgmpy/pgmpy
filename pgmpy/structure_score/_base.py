from __future__ import annotations

from functools import lru_cache

import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.utils import build_state_names, get_dataset_type, preprocess_data


class BaseStructureScore(BaseObject):
    """
    Abstract base class for structure scoring in pgmpy.

    Structure scores evaluate how well a candidate Bayesian network structure fits observed
    data. This class implements the shared scoring workflow, caching for local scores, and
    utilities for computing conditional state counts. Use one of the concrete score classes
    such as `K2`, `BDeu`, `BIC`, or `AIC` instead of instantiating this class directly.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame in which each column represents a variable. Missing values should be marked
        as `numpy.nan`.
    state_names : dict, optional
        Dictionary mapping each variable name to its allowed states. If not specified, the
        observed values in the data are used.
    max_cache_size : int or None, default=10000
        Maximum number of local scores to cache. If None, the cache is unlimited.
        Increase this for large datasets to avoid cache thrashing.
    """

    _tags = {
        "name": None,
        "supported_datatype": None,
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None, max_cache_size=10000):
        self.data, self.dtypes = preprocess_data(data)
        self.cache_size = max_cache_size
        if max_cache_size is not None and max_cache_size <= 0:
            raise ValueError(f"cache_size must be a positive integer or None. Got: {max_cache_size}")

        if self.data is not None:
            self.variables = list(self.data.columns.values)
            self.state_names = build_state_names(self.data, state_names=state_names)

        self._cached_local_score = lru_cache(maxsize=max_cache_size)(self._local_score)

    def local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """Compute the cached local score for `variable` given `parents`."""
        return self._cached_local_score(variable, parents)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """Compute the uncached local score for `variable` given `parents`."""
        raise NotImplementedError

    def score(self, model) -> float:
        """Compute a structure score for a model."""
        score = 0
        for node in model.nodes():
            score += self.local_score(node, tuple(model.predecessors(node)))
        score += self.structure_prior(model)
        return score

    def structure_prior(self, model) -> float:
        """Return the log prior over structures."""
        return 0

    def structure_prior_ratio(self, operation) -> float:
        """Return the log prior ratio for a structure operation."""
        return 0


def get_scoring_method(
    scoring_method: str | BaseStructureScore | None,
    data: pd.DataFrame,
) -> BaseStructureScore:
    """
    Returns a structure score instance.

    Parameters
    ----------
    scoring_method : str or BaseStructureScore or None
        The scoring method whose instance is to be returned.

        - If a string is provided, the corresponding scoring method is
        instantiated with default parameters.
        - If a ``BaseStructureScore`` instance is provided, it is returned
        unchanged.
        - If ``None``, the default scoring method for the data type is
        selected automatically.

    data : pandas.DataFrame
        Dataset used to determine the default scoring method and to
        initialize score instances.

    Returns
    -------
    BaseStructureScore
        An initialized structure score instance.

    Examples
    --------
    >>> from pgmpy.example_models import load_model
    >>> from pgmpy.structure_score import BDeu, get_scoring_method
    >>> model = load_model("bnlearn/asia")
    >>> data = model.simulate(n_samples=1000, seed=42)

    Use a scoring method by name:

    >>> score = get_scoring_method("k2", data)

    Use the default scoring method for the dataset type:

    >>> score = get_scoring_method(None, data)

    Use a custom scoring method with non-default parameters:

    >>> score = BDeu(data, equivalent_sample_size=20, max_cache_size=20000)
    >>> score = get_scoring_method(score, data)

    Raises
    ------
    ValueError
        If ``scoring_method`` is invalid, unknown, or if ``data`` is
        required but not provided.
    """
    if isinstance(scoring_method, BaseStructureScore):
        return scoring_method

    if scoring_method is None:
        if data is None:
            raise ValueError("Cannot determine scoring method: both `scoring_method` and `data` are None.")
        var_type = get_dataset_type(data)
        filter_tags = {"default_for": var_type}
    elif isinstance(scoring_method, str):
        filter_tags = {"name": scoring_method.lower()}
    else:
        raise ValueError(f"Invalid `scoring_method` argument: {scoring_method!r}")

    scores = all_objects(
        object_types=BaseStructureScore,
        package_name="pgmpy.structure_score",
        return_names=False,
        filter_tags=filter_tags,
    )

    if scores:
        cls = scores[0]
        if data is None:
            raise ValueError(f"Scoring method '{cls.__name__}' requires data, but data is None.")

        return cls(data=data)

    else:
        raise ValueError(f"Unknown scoring method: {scoring_method!r}")
