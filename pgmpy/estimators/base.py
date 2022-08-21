#!/usr/bin/env python
from functools import lru_cache

import pandas as pd

from pgmpy.utils.decorators import convert_args_tuple


class BaseEstimator(object):
    def __init__(self, data=None, state_names=None, complete_samples_only=True):
        """
        Base class for estimators in pgmpy; `ParameterEstimator`,
        `StructureEstimator` and `StructureScore` derive from this class.

        Parameters
        ----------
        data: pandas DataFrame object
            object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        """

        self.data = data
        # data can be None in the case when learning structure from
        # independence conditions. Look into PC.py.
        if self.data is not None:
            self.complete_samples_only = complete_samples_only

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

    @convert_args_tuple
    @lru_cache(maxsize=2048)
    def state_counts(
        self, variable, parents=[], complete_samples_only=None, weighted=False
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

        complete_samples_only: bool
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.NaN` somewhere are ignored. If `False` then
            every row where neither the variable nor its parents are `np.NaN` is used.
            Desired default behavior can be passed to the class constructor.

        weighted: bool
            If True, data must have a `_weight` column specifying the weight of the
            datapoint (row). If False, each datapoint has a weight of `1`.

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

        # default for how to deal with missing data can be set in class constructor
        if complete_samples_only is None:
            complete_samples_only = self.complete_samples_only
        # ignores either any row containing NaN, or only those where the variable or its parents is NaN
        data = (
            self.data.dropna()
            if complete_samples_only
            else self.data.dropna(subset=[variable] + parents)
        )

        if weighted and ("_weight" not in self.data.columns):
            raise ValueError("data must contain a `_weight` column if weighted=True")

        if not parents:
            # count how often each state of 'variable' occurred
            if weighted:
                state_count_data = data.groupby([variable]).sum()["_weight"]
            else:
                state_count_data = data.loc[:, variable].value_counts()

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
                    data.groupby([variable] + parents).sum()["_weight"].unstack(parents)
                )

            else:
                state_count_data = (
                    data.groupby([variable] + parents).size().unstack(parents)
                )

            if not isinstance(state_count_data.columns, pd.MultiIndex):
                state_count_data.columns = pd.MultiIndex.from_arrays(
                    [state_count_data.columns]
                )

            # reindex rows & columns to sort them and to add missing ones
            # missing row    = some state of 'variable' did not occur in data
            # missing column = some state configuration of current 'variable's parents
            #                  did not occur in data
            row_index = self.state_names[variable]
            column_index = pd.MultiIndex.from_product(parents_states, names=parents)
            state_counts = state_count_data.reindex(
                index=row_index, columns=column_index
            ).fillna(0)

        return state_counts


class ParameterEstimator(BaseEstimator):
    def __init__(self, model, data, **kwargs):
        """
        Base class for parameter estimators in pgmpy.

        Parameters
        ----------
        model: pgmpy.models.BayesianNetwork or pgmpy.models.MarkovNetwork or pgmpy.models.NoisyOrModel model
            for which parameter estimation is to be done.

        data: pandas DataFrame object
            dataframe object with column names identical to the variable names of the model.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        """

        if not (set(model.nodes()) - model.latents) <= set(data.columns.values):
            raise ValueError(
                "variable names of the model must be identical to column names in data"
            )
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

        complete_samples_only: bool
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.NaN` somewhere are ignored. If `False` then
            every row where neither the variable nor its parents are `np.NaN` is used.
            Desired default behavior can be passed to the class constructor.

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
    def __init__(self, data=None, independencies=None, **kwargs):
        """
        Base class for structure estimators in pgmpy.

        Parameters
        ----------
        data: pandas DataFrame object
            dataframe object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        """

        self.independencies = independencies
        if self.independencies is not None:
            self.variables = self.independencies.get_all_variables()

        super(StructureEstimator, self).__init__(data=data, **kwargs)

    def estimate(self):
        pass
