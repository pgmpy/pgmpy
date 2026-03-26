from __future__ import annotations

from functools import lru_cache

import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.utils import build_state_names, get_dataset_type, get_state_counts, preprocess_data


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
    """

    _tags = {
        "name": None,
        "supported_datatype": None,
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        self.data, self.dtypes = preprocess_data(data)

        if self.data is not None:
            self.variables = list(self.data.columns.values)
            self.state_names = build_state_names(self.data, state_names=state_names)

        self._cached_local_score = lru_cache(maxsize=10000)(self._local_score)

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

    def state_counts(
        self,
        variable: str,
        parents: tuple[str, ...] = (),
        weighted: bool = False,
        reindex: bool = True,
    ) -> pd.DataFrame:
        """Return state counts for `variable`, optionally conditioned on `parents`."""
        return get_state_counts(
            data=self.data,
            state_names=self.state_names,
            variable=variable,
            parents=parents,
            weighted=weighted,
            reindex=reindex,
        )


def get_scoring_method(
    scoring_method: str | BaseStructureScore | None,
    data: pd.DataFrame,
) -> BaseStructureScore:
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
