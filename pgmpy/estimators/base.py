#!/usr/bin/env python

from collections import defaultdict
import numpy as np
import pandas as pd
import warnings

from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference.ExactInference import BeliefPropagation
from pgmpy.utils import preprocess_data


class BaseEstimator(object):
    """
    Base class for estimators in pgmpy; `ParameterEstimator`,
    `StructureEstimator` and `StructureScore` derive from this class.
    """

    def __init__(self, data=None, state_names=None, **init_params):
        if data is None:
            self.data = None
            self.dtypes = None
        else:
            self.data, self.dtypes = preprocess_data(data)

        self._init_params = init_params
        self.state_names = state_names or {}

        if self.data is not None:
            self.variables = list(self.data.columns.values)

            if not isinstance(self.state_names, dict):
                self.state_names = {
                    var: self._collect_state_names(var) for var in self.variables
                }
            else:
                for var in self.variables:
                    if var in self.state_names:
                        if not set(self._collect_state_names(var)) <= set(
                            self.state_names[var]
                        ):
                            raise ValueError(
                                f"Data contains unexpected states for variable: {var}."
                            )
                    else:
                        self.state_names[var] = self._collect_state_names(var)

    def estimate(self, data=None, **params):
        if data is None:
            data = self.data
        else:
            data, _ = preprocess_data(data)

        if data is None:
            raise ValueError("Data must be provided either during __init__ or in estimate().")

        merged_params = {**self._init_params, **params}
        unknown = set(params) - set(self._init_params)

        if unknown:
            warnings.warn(
                f"Passing parameters to estimate() directly is deprecated and will be removed in v2. "
                f"Please pass these to __init__ instead: {unknown}",
                DeprecationWarning,
                stacklevel=2
            )

        return self._estimate(data, **merged_params)

    def _estimate(self, data, **params):
        raise NotImplementedError("Subclasses must implement _estimate(data, **params).")

    def _collect_state_names(self, variable):
        return sorted(list(self.data.loc[:, variable].dropna().unique()))

    def state_counts(self, variable, parents=[], weighted=False, reindex=True):
        parents = list(parents)

        if weighted and "_weight" not in self.data.columns:
            raise ValueError("data must contain a `_weight` column if weighted=True")

        if not parents:
            if weighted:
                state_count_data = self.data.groupby([variable], observed=True)["_weight"].sum()
            else:
                state_count_data = self.data.loc[:, variable].value_counts()

            state_counts = (
                state_count_data.reindex(self.state_names[variable])
                .fillna(0)
                .to_frame()
            )
        else:
            parents_states = [self.state_names[parent] for parent in parents]

            if weighted:
                state_count_data = (
                    self.data.groupby([variable] + parents, observed=True)["_weight"]
                    .sum()
                    .unstack(parents)
                )
            else:
                state_count_data = (
                    self.data.groupby([variable] + parents, observed=True)
                    .size()
                    .unstack(parents)
                )

            if not isinstance(state_count_data.columns, pd.MultiIndex):
                state_count_data.columns = pd.MultiIndex.from_arrays(
                    [state_count_data.columns]
                )

            if reindex:
                row_index = self.state_names[variable]
                column_index = pd.MultiIndex.from_product(parents_states, names=parents)
                state_counts = state_count_data.reindex(
                    index=row_index, columns=column_index
                ).fillna(0)
            else:
                state_counts = state_count_data.fillna(0)

        return state_counts


class ParameterEstimator(BaseEstimator):
    def __init__(self, model, data=None, **kwargs):
        self.model = model
        super(ParameterEstimator, self).__init__(data, **kwargs)

    def _estimate(self, data, **kwargs):
        raise NotImplementedError("ParameterEstimator subclasses must implement _estimate.")

    def state_counts(self, variable, weighted=False, **kwargs):
        parents = sorted(self.model.get_parents(variable))
        return super(ParameterEstimator, self).state_counts(
            variable, parents=parents, weighted=weighted, **kwargs
        )


class StructureEstimator(BaseEstimator):
    def __init__(self, data=None, independencies=None, **kwargs):
        self.independencies = independencies
        if self.independencies is not None:
            self.variables = self.independencies.get_all_variables()

        super(StructureEstimator, self).__init__(data=data, **kwargs)

    def _estimate(self, data, **params):
        raise NotImplementedError("StructureEstimator subclasses must implement _estimate.")


class MarginalEstimator(BaseEstimator):
    def __init__(self, model, data=None, **kwargs):
        super().__init__(data, **kwargs)
        self.belief_propagation = BeliefPropagation(model=model)
        self.theta = None

    @staticmethod
    def _clique_to_marginal(marginals, clique_nodes):
        clique_to_marginal = defaultdict(lambda: [])
        for marginal_clique, marginal in marginals.items():
            for clique in clique_nodes:
                if set(marginal_clique) <= set(clique):
                    clique_to_marginal[clique].append(marginal)
                    break
            else:
                raise ValueError(
                    f"Could not find a corresponding clique for marginal: {marginal_clique}"
                    f" out of cliques: {clique_nodes}"
                )
        return clique_to_marginal

    def _marginal_loss(self, marginals, clique_to_marginal, metric):
        loss = 0.0
        gradient = FactorDict({})

        for clique, mu in marginals.items():
            gradient[clique] = mu.identity_factor() * 0
            for y in clique_to_marginal[clique]:
                projection_variables = list(set(mu.scope()) - set(y.scope()))
                mu2 = mu.marginalize(variables=projection_variables, inplace=False)

                if not isinstance(mu2, DiscreteFactor):
                    raise TypeError(f"Expected DiscreteFactor, got {type(mu2)}")

                diff_factor = mu2 + (y * -1)
                if not diff_factor:
                    raise ValueError("Error when calculating gradient.")

                diff = diff_factor.values.flatten()

                if metric == "L1":
                    loss += abs(diff).sum()
                    grad = diff.sign() if hasattr(diff, "sign") else np.sign(diff)
                elif metric == "L2":
                    loss += 0.5 * (diff @ diff)
                    grad = diff
                else:
                    raise ValueError("Metric must be one of L1 or L2.")

                gradient[clique] += DiscreteFactor(
                    variables=mu2.scope(),
                    cardinality=mu2.cardinality,
                    values=grad,
                    state_names=mu2.state_names,
                )

        return loss, gradient

    def _estimate(self, data, **params):
        raise NotImplementedError("MarginalEstimator subclasses must implement _estimate.")
