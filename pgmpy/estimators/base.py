#!/usr/bin/env python

from collections import defaultdict

import numpy as np
import pandas as pd

from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference.ExactInference import BeliefPropagation


class BaseEstimator(object):
    """
    Base class for estimators in pgmpy; `ParameterEstimator`,
    `StructureEstimator` and `StructureScore` derive from this class.

    Parameters
    ----------
    data: pandas DataFrame object
        object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.
    """

    def __init__(self, data=None, state_names=None):
        self.data = data
        # data can be None in the case when learning structure from
        # independence conditions. Look into PC.py.
        if self.data is not None:
            self.variables = list(data.columns.values)

            if not isinstance(state_names, dict):
                self.state_names = {
                    var: self._collect_state_names(var) for var in self.variables
                }
            else:
                self.state_names = dict()
                for var in self.variables:
                    if var in state_names:
                        if not set(self._collect_state_names(var)) <= set(
                            state_names[var]
                        ):
                            raise ValueError(
                                f"Data contains unexpected states for variable: {var}."
                            )
                        self.state_names[var] = state_names[var]
                    else:
                        self.state_names[var] = self._collect_state_names(var)

    def _collect_state_names(self, variable):
        "Return a list of states that the variable takes in the data."
        states = sorted(list(self.data.loc[:, variable].dropna().unique()))
        return states

    def state_counts(
        self,
        variable,
        parents=[],
        weighted=False,
        reindex=True,
    ):
        """
        Return counts how often each state of 'variable' occurred in the data.
        If a list of parents is provided, counting is done conditionally
        for each state configuration of the parents.

        Parameters
        ----------
        variable: string
            Name of the variable for which the state count is to be done.

        parents: list
            Optional list of variable parents, if conditional counting is desired.
            Order of parents in list is reflected in the returned DataFrame

        weighted: bool
            If True, data must have a `_weight` column specifying the weight of the
            datapoint (row). If False, each datapoint has a weight of `1`.

        reindex: bool
            If True, returns a data frame with all possible parents state combinations
            as the columns. If False, drops the state combinations which are not
            present in the data.

        Returns
        -------
        state_counts: pandas.DataFrame
            Table with state counts for 'variable'

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.estimators import BaseEstimator
        >>> data = pd.DataFrame(data={'A': ['a1', 'a1', 'a2'],
                                      'B': ['b1', 'b2', 'b1'],
                                      'C': ['c1', 'c1', 'c2']})
        >>> estimator = BaseEstimator(data)
        >>> estimator.state_counts('A')
            A
        a1  2
        a2  1
        >>> estimator.state_counts('C', parents=['A', 'B'])
        A  a1      a2
        B  b1  b2  b1  b2
        C
        c1  1   1   0   0
        c2  0   0   1   0
        >>> estimator.state_counts('C', parents=['A'])
        A    a1   a2
        C
        c1  2.0  0.0
        c2  0.0  1.0
        """
        parents = list(parents)

        if weighted and ("_weight" not in self.data.columns):
            raise ValueError("data must contain a `_weight` column if weighted=True")

        if not parents:
            # count how often each state of 'variable' occurred
            if weighted:
                state_count_data = self.data.groupby([variable])["_weight"].sum()
            else:
                state_count_data = self.data.loc[:, variable].value_counts()

            state_counts = (
                state_count_data.reindex(self.state_names[variable])
                .fillna(0)
                .to_frame()
            )

        else:
            parents_states = [self.state_names[parent] for parent in parents]
            # count how often each state of 'variable' occurred, conditional on parents' states
            if weighted:
                state_count_data = (
                    self.data.groupby([variable] + parents)["_weight"]
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
                # reindex rows & columns to sort them and to add missing ones
                # missing row    = some state of 'variable' did not occur in data
                # missing column = some state configuration of current 'variable's parents
                #                  did not occur in data
                row_index = self.state_names[variable]
                column_index = pd.MultiIndex.from_product(parents_states, names=parents)
                state_counts = state_count_data.reindex(
                    index=row_index, columns=column_index
                ).fillna(0)
            else:
                state_counts = state_count_data.fillna(0)

        return state_counts


class ParameterEstimator(BaseEstimator):
    """
    Base class for parameter estimators in pgmpy.

    Parameters
    ----------
    model: pgmpy.models.BayesianNetwork or pgmpy.models.MarkovNetwork or pgmpy.models.NoisyOrModel model
        for which parameter estimation is to be done.

    data: pandas DataFrame object
        dataframe object with column names identical to the variable names of the model.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.
    """

    def __init__(self, model, data, **kwargs):
        """
        Base class for parameter estimators in pgmpy.

        Parameters
        ----------
        model: pgmpy.models.BayesianNetwork or pgmpy.models.MarkovNetwork or pgmpy.models.NoisyOrModel model
            for which parameter estimation is to be done.

        data: pandas DataFrame object
            dataframe object with column names identical to the variable names of the model.
            (If some values in the data are missing the data cells should be set to `numpy.nan`.
            Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.nan` is used.
            This sets the behavior of the `state_count`-method.
        """
        self.model = model

        super(ParameterEstimator, self).__init__(data, **kwargs)

    def state_counts(self, variable, weighted=False, **kwargs):
        """
        Return counts how often each state of 'variable' occurred in the data.
        If the variable has parents, counting is done conditionally
        for each state configuration of the parents.

        Parameters
        ----------
        variable: string
            Name of the variable for which the state count is to be done.

        Returns
        -------
        state_counts: pandas.DataFrame
            Table with state counts for 'variable'

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.estimators import ParameterEstimator
        >>> model = BayesianNetwork([('A', 'C'), ('B', 'C')])
        >>> data = pd.DataFrame(data={'A': ['a1', 'a1', 'a2'],
                                      'B': ['b1', 'b2', 'b1'],
                                      'C': ['c1', 'c1', 'c2']})
        >>> estimator = ParameterEstimator(model, data)
        >>> estimator.state_counts('A')
            A
        a1  2
        a2  1
        >>> estimator.state_counts('C')
        A  a1      a2
        B  b1  b2  b1  b2
        C
        c1  1   1   0   0
        c2  0   0   1   0
        """

        parents = sorted(self.model.get_parents(variable))
        return super(ParameterEstimator, self).state_counts(
            variable, parents=parents, weighted=weighted, **kwargs
        )


class StructureEstimator(BaseEstimator):
    """
    Base class for structure estimators in pgmpy.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.
    """

    def __init__(self, data=None, independencies=None, **kwargs):
        self.independencies = independencies
        if self.independencies is not None:
            self.variables = self.independencies.get_all_variables()

        super(StructureEstimator, self).__init__(data=data, **kwargs)

    def estimate(self):
        pass


class MarginalEstimator(BaseEstimator):
    """
    Base class for marginal estimators in pgmpy.

    Parameters
    ----------
    model: MarkovNetwork | FactorGraph | JunctionTree
        A model to optimize, using Belief Propagation and an estimation method.

    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.
    """

    def __init__(self, model, data, **kwargs):
        super().__init__(data, **kwargs)
        self.belief_propagation = BeliefPropagation(model=model)
        self.theta = None

    @staticmethod
    def _clique_to_marginal(marginals, clique_nodes):
        """
        Construct a minimal mapping from cliques to marginals.

        Parameters
        ----------
        marginals: FactorDict
            A mapping from cliques to factors.

        clique_nodes: List[Tuple[str, ...]]
            Cliques that exist within a different FactorDict.

        Returns
        -------
        clique_to_marginal: A mapping from clique to a list of marginals
        such that each clique is a super set of the marginals it is associated with.
        """
        clique_to_marginal = defaultdict(lambda: [])
        for marginal_clique, marginal in marginals.items():
            for clique in clique_nodes:
                if set(marginal_clique) <= set(clique):
                    clique_to_marginal[clique].append(marginal)
                    break
            else:
                raise ValueError(
                    "Could not find a corresponding clique for"
                    + f" marginal: {marginal_clique}"
                    + f" out of cliques: {clique_nodes}"
                )
        return clique_to_marginal

    def _marginal_loss(self, marginals, clique_to_marginal, metric):
        """
        Compute the loss and gradient for a given dictionary of clique beliefs.

        Parameters
        ----------
        marginals: FactorDict
            A mapping from a clique to an observed marginal represented by a `DiscreteFactor`.

        clique_to_marginal: Dict[Tuple[str, ...], List[DiscreteFactor]]
            A mapping from a Junction Tree's clique to a list of corresponding marginals
            such that a clique is a superset of the marginal with the constraint that
            each marginal only appears once across all cliques.

        metric: str
            One of either 'L1' or 'L2'.

        Returns
        -------
        Loss and gradient of the loss: Tuple[float, pgmpy.factors.FactorDict.FactorDict]
            Marginal loss and the gradients of the loss with respect to the estimated beliefs.
        """
        loss = 0.0
        gradient = FactorDict({})

        for clique, mu in marginals.items():
            # Initialize a gradient for this clique as zero.
            gradient[clique] = mu.identity_factor() * 0

            # Iterate over all marginals involving this clique.
            for y in clique_to_marginal[clique]:
                # Step 1: Marginalize the clique to the size of `y`.
                projection_variables = list(set(mu.scope()) - set(y.scope()))
                mu2 = mu.marginalize(
                    variables=projection_variables,
                    inplace=False,
                )

                if not isinstance(mu2, DiscreteFactor):
                    raise TypeError(f"Expecting a DiscreteFactor but found {type(mu2)}")

                # Step 2: Compute the difference between the `mu2` and `y`.
                diff_factor = mu2 + (y * -1)

                if not diff_factor:
                    raise ValueError("An error occured when calculating the gradient.")

                diff = diff_factor.values.flatten()

                # Step 3: Compute the loss and gradient based upon the metric.
                if metric == "L1":
                    loss += abs(diff).sum()
                    grad = diff.sign() if hasattr(diff, "sign") else np.sign(diff)
                elif metric == "L2":
                    loss += 0.5 * (diff @ diff)
                    grad = diff
                else:
                    raise ValueError("Metric must be one of L1 or L2.")

                # Step 4: Update the gradient from this marginal.
                gradient[clique] += DiscreteFactor(
                    variables=mu2.scope(),
                    cardinality=mu2.cardinality,
                    values=grad,
                    state_names=mu2.state_names,
                )

        return loss, gradient

    def estimate(self):
        pass
