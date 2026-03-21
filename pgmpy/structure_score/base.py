import pandas as pd
from skbase.base import BaseObject

from pgmpy.utils import preprocess_data


class BaseStructureScore(BaseObject):
    """Base class for structure scoring."""

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

    def score(self, model) -> float:
        """Compute a structure score for a model."""
        score = 0
        for node in model.nodes():
            score += self.local_score(node, list(model.predecessors(node)))
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
        parents=[],
        weighted: bool = False,
        reindex: bool = True,
    ) -> pd.DataFrame:
        """Return state counts for `variable`, optionally conditioned on `parents`."""
        parents = list(parents)

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
                    self.data.groupby([variable] + parents, observed=True)["_weight"].sum().unstack(parents)
                )
            else:
                state_count_data = self.data.groupby([variable] + parents, observed=True).size().unstack(parents)

            if not isinstance(state_count_data.columns, pd.MultiIndex):
                state_count_data.columns = pd.MultiIndex.from_arrays([state_count_data.columns])

            if reindex:
                row_index = self.state_names[variable]
                column_index = pd.MultiIndex.from_product(parents_states, names=parents)
                state_counts = state_count_data.reindex(index=row_index, columns=column_index).fillna(0)
            else:
                state_counts = state_count_data.fillna(0)

        return state_counts
