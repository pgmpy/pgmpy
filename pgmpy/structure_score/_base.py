from __future__ import annotations

from functools import lru_cache

import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.utils import get_dataset_type, preprocess_data


class BaseStructureScore(BaseObject):
    """Base class for structure scoring."""

    _tags = {
        "name": None,
        "supported_datatype": None,
        "default_for": None,
        "requires_data": True,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None, **kwargs):
        self.data, self.dtypes = preprocess_data(data)

        if self.data is not None:
            self.variables = list(data.columns.values)

            if not isinstance(state_names, dict):
                self.state_names = {var: self._collect_state_names(var) for var in self.variables}
            else:
                self.state_names = dict()
                for var in self.variables:
                    if var in state_names:
                        if not set(self._collect_state_names(var)) <= set(state_names[var]):
                            raise ValueError(f"Data contains unexpected states for variable: {var}.")
                        self.state_names[var] = state_names[var]
                    else:
                        self.state_names[var] = self._collect_state_names(var)

    def _collect_state_names(self, variable: str) -> list:
        """Return a list of states that the variable takes in the data."""
        states = sorted(list(self.data.loc[:, variable].dropna().unique()))
        return states

    @staticmethod
    def _validate_parents(parents: tuple[str, ...]) -> tuple[str, ...]:
        """Validate that parent variables are provided as a tuple."""
        if not isinstance(parents, tuple):
            raise TypeError("`parents` must be a tuple.")
        return parents

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
        parents = self._validate_parents(parents)
        parent_list = list(parents)

        if weighted and ("_weight" not in self.data.columns):
            raise ValueError("data must contain a `_weight` column if weighted=True")

        if not parents:
            if weighted:
                state_count_data = self.data.groupby([variable], observed=True)["_weight"].sum()
            else:
                state_count_data = self.data.loc[:, variable].value_counts()

            state_counts = state_count_data.reindex(self.state_names[variable]).fillna(0).to_frame()

        else:
            parents_states = [self.state_names[parent] for parent in parents]
            if weighted:
                state_count_data = (
                    self.data.groupby([variable] + parent_list, observed=True)["_weight"].sum().unstack(parent_list)
                )
            else:
                state_count_data = (
                    self.data.groupby([variable] + parent_list, observed=True).size().unstack(parent_list)
                )

            if not isinstance(state_count_data.columns, pd.MultiIndex):
                state_count_data.columns = pd.MultiIndex.from_arrays([state_count_data.columns])

            if reindex:
                row_index = self.state_names[variable]
                column_index = pd.MultiIndex.from_product(parents_states, names=parent_list)
                state_counts = state_count_data.reindex(index=row_index, columns=column_index).fillna(0)
            else:
                state_counts = state_count_data.fillna(0)

        return state_counts


def _enable_local_score_cache(score: BaseStructureScore, max_size: int = 10000) -> BaseStructureScore:
    if not isinstance(score, BaseStructureScore):
        raise TypeError("`score` must be an instance of BaseStructureScore.")

    if not getattr(score, "_local_score_cache_enabled", False):
        score.local_score = lru_cache(maxsize=int(max_size))(score.local_score)
        score._local_score_cache_enabled = True

    return score


def get_scoring_method(
    scoring_method: str | BaseStructureScore | None,
    data: pd.DataFrame,
    use_cache: bool = True,
    **kwargs,
) -> tuple[BaseStructureScore, BaseStructureScore]:
    if isinstance(scoring_method, BaseStructureScore):
        if use_cache:
            scoring_method = _enable_local_score_cache(scoring_method)
        return scoring_method, scoring_method

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
        if cls.get_class_tag("requires_data", tag_value_default=True):
            if data is None:
                raise ValueError(f"Scoring method '{cls.__name__}' requires data, but data is None.")
            score = cls(data=data, **kwargs)
        else:
            score = cls(**kwargs)
        if use_cache:
            score = _enable_local_score_cache(score)
        return score, score

    raise ValueError(f"Unknown scoring method: {scoring_method!r}")
