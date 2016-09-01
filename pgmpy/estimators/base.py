#!/usr/bin/env python
from warnings import warn

import numpy as np
import pandas as pd
from scipy.stats import chisquare


class BaseEstimator(object):
    def __init__(self, data, state_names=None, complete_samples_only=True):
        """
        Base class for estimators in pgmpy; `ParameterEstimator`,
        `StructureEstimator` and `StructureScore` derive from this class.

        Parameters
        ----------

        data: pandas DataFrame object
            datafame object where each column represents one variable.
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
        self.complete_samples_only = complete_samples_only

        variables = list(data.columns.values)

        if not isinstance(state_names, dict):
            self.state_names = {var: self._collect_state_names(var) for var in variables}
        else:
            self.state_names = dict()
            for var in variables:
                if var in state_names:
                    if not set(self._collect_state_names(var)) <= set(state_names[var]):
                        raise ValueError("Data contains unexpected states for variable '{0}'.".format(str(var)))
                    self.state_names[var] = sorted(state_names[var])
                else:
                    self.state_names[var] = self._collect_state_names(var)

    def _collect_state_names(self, variable):
        "Return a list of states that the variable takes in the data"
        states = sorted(list(self.data.ix[:, variable].dropna().unique()))
        return states

    def state_counts(self, variable, parents=[], complete_samples_only=None):
        """
        Return counts how often each state of 'variable' occured in the data.
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
        """

        # default for how to deal with missing data can be set in class constructor
        if complete_samples_only is None:
            complete_samples_only = self.complete_samples_only
        # ignores either any row containing NaN, or only those where the variable or its parents is NaN
        data = self.data.dropna() if complete_samples_only else self.data.dropna(subset=[variable] + parents)

        if not parents:
            # count how often each state of 'variable' occured
            state_count_data = data.ix[:, variable].value_counts()
            state_counts = state_count_data.reindex(self.state_names[variable]).fillna(0).to_frame()

        else:
            parents_states = [self.state_names[parent] for parent in parents]
            # count how often each state of 'variable' occured, conditional on parents' states
            state_count_data = data.groupby([variable] + parents).size().unstack(parents)

            # reindex rows & columns to sort them and to add missing ones
            # missing row    = some state of 'variable' did not occur in data
            # missing column = some state configuration of current 'variable's parents
            #                  did not occur in data
            row_index = self.state_names[variable]
            column_index = pd.MultiIndex.from_product(parents_states, names=parents)
            state_counts = state_count_data.reindex(index=row_index, columns=column_index).fillna(0)

        return state_counts

    def test_conditional_independence(self, X, Y, Zs=[]):
        """Chi-square conditional independence test.
        Tests the null hypothesis that X is independent from Y given Zs.

        This is done by comparing the observed frequencies with the expected
        frequencies if X,Y were conditionally independent, using a chisquare
        deviance statistic. The expected frequencies given independence are
        `P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
        as `P(X,Zs)*P(Y,Zs)/P(Zs).

        Parameters
        ----------
        X: int, string, hashable object
            A variable name contained in the data set
        Y: int, string, hashable object
            A variable name contained in the data set, different from X
        Zs: list of variable names
            A list of variable names contained in the data set, different from X and Y.
            This is the separating set that (potentially) makes X and Y independent.
            Default: []

        Returns
        -------
        chi2: float
            The chi2 test statistic.
        p_value: float
            The p_value, i.e. the probability of observing the computed chi2
            statistic (or an even higher value), given the null hypothesis
            that X _|_ Y | Zs.
        sufficient_data: bool
            A flag that indicates if the sample size is considered sufficient.
            As in [4], require at least 5 samples per parameter (on average).
            That is, the size of the data set must be greater than
            `5 * (c(X) - 1) * (c(Y) - 1) * prod([c(Z) for Z in Zs])`
            (c() denotes the variable cardinality).


        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.2.2.3 (page 789)
        [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence
        [4] Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
        >>> data['E'] = data['A'] + data['B'] + data['C']
        >>> c = ConstraintBasedEstimator(data)
        >>> print(c.test_conditional_independence('A', 'C'))  # independent
        (0.95035644482050263, 0.8132617142699442, True)
        >>> print(c.test_conditional_independence('A', 'B', 'D'))  # independent
        (5.5227461320130899, 0.59644169242588885, True)
        >>> print(c.test_conditional_independence('A', 'B', ['D', 'E']))  # dependent
        (9192.5172226063387, 0.0, True)
        """

        if isinstance(Zs, (frozenset, list, set, tuple,)):
            Zs = list(Zs)
        else:
            Zs = [Zs]

        num_params = ((len(self.state_names[X])-1) *
                      (len(self.state_names[Y])-1) *
                      np.prod([len(self.state_names[Z]) for Z in Zs]))
        sufficient_data = len(self.data) >= num_params * 5
        if not sufficient_data:
            warn("Insufficient data for testing {0} _|_ {1} | {2}. ".format(X, Y, Zs) +
                 "At least {0} samples recommended, {1} present.".format(5 * num_params, len(self.data)))

        # compute actual frequency/state_count table:
        # = P(X,Y,Zs)
        XYZ_state_counts = pd.crosstab(index=self.data[X],
                                       columns=[self.data[Y]] + [self.data[Z] for Z in Zs])
        # reindex to add missing rows & columns (if some values don't appear in data)
        row_index = self.state_names[X]
        column_index = pd.MultiIndex.from_product(
                            [self.state_names[Y]] + [self.state_names[Z] for Z in Zs], names=[Y]+Zs)
        XYZ_state_counts = XYZ_state_counts.reindex(index=row_index,    columns=column_index).fillna(0)

        # compute the expected frequency/state_count table if X _|_ Y | Zs:
        # = P(X|Zs)*P(Y|Zs)*P(Zs) = P(X,Zs)*P(Y,Zs)/P(Zs)
        if Zs:
            XZ_state_counts = XYZ_state_counts.sum(axis=1, level=Zs)  # marginalize out Y
            YZ_state_counts = XYZ_state_counts.sum().unstack(Zs)      # marginalize out X
        else:
            XZ_state_counts = XYZ_state_counts.sum(axis=1)
            YZ_state_counts = XYZ_state_counts.sum()
        Z_state_counts = YZ_state_counts.sum()  # marginalize out both

        XYZ_expected = pd.DataFrame(index=XYZ_state_counts.index, columns=XYZ_state_counts.columns)
        for X_val in XYZ_expected.index:
            if Zs:
                for Y_val in XYZ_expected.columns.levels[0]:
                    XYZ_expected.loc[X_val, Y_val] = (XZ_state_counts.loc[X_val] *
                                                      YZ_state_counts.loc[Y_val] /
                                                      Z_state_counts).values
            else:
                for Y_val in XYZ_expected.columns:
                    XYZ_expected.loc[X_val, Y_val] = (XZ_state_counts.loc[X_val] *
                                                      YZ_state_counts.loc[Y_val] /
                                                      float(Z_state_counts))

        observed = XYZ_state_counts.values.flatten()
        expected = XYZ_expected.fillna(0).values.flatten()
        # remove elements where the expected value is 0;
        # this also corrects the degrees of freedom for chisquare
        observed, expected = zip(*((o, e) for o, e in zip(observed, expected) if not e == 0))

        chi2, significance_level = chisquare(observed, expected)

        return (chi2, significance_level, sufficient_data)


class ParameterEstimator(BaseEstimator):
    def __init__(self, model, data, **kwargs):
        """
        Base class for parameter estimators in pgmpy.

        Parameters
        ----------
        model: pgmpy.models.BayesianModel or pgmpy.models.MarkovModel or pgmpy.models.NoisyOrModel
            model for which parameter estimation is to be done

        data: pandas DataFrame object
            datafame object with column names identical to the variable names of the model.
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

        if not set(model.nodes()) <= set(data.columns.values):
            raise ValueError("variable names of the model must be identical to column names in data")
        self.model = model

        super(ParameterEstimator, self).__init__(data, **kwargs)

    def state_counts(self, variable, **kwargs):
        """
        Return counts how often each state of 'variable' occured in the data.
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
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import ParameterEstimator
        >>> model = BayesianModel([('A', 'C'), ('B', 'C')])
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
        return super(ParameterEstimator, self).state_counts(variable, parents=parents, **kwargs)

    def get_parameters(self):
        pass


class StructureEstimator(BaseEstimator):
    def __init__(self, data, **kwargs):
        """
        Base class for structure estimators in pgmpy.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
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

        super(StructureEstimator, self).__init__(data, **kwargs)

    def estimate(self):
        pass
