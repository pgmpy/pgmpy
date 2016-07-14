#!/usr/bin/env python
import numpy as np
import pandas as pd
from warnings import warn
from itertools import combinations
from scipy.stats import chisquare

from pgmpy.base import UndirectedGraph
from pgmpy.models import BayesianModel
from pgmpy.estimators import StructureEstimator


class ConstraintBasedEstimator(StructureEstimator):
    def __init__(self, data, **kwargs):
        """
        Class for constraint-based estimation of BayesianModels from a given
        data set. Identifies (conditional) dependencies in data set using
        chi_square dependency test and uses the PC algorithm to estimate a DAG
        pattern that satisfies the identified dependencies. The DAG pattern can
        then be completed to a faithful BayesianModel, if possible.

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

        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques,
            2009, Section 18.2
        [2] Neapolitan, Learning Bayesian Networks, Section 10.1.2 for the PC algorithm (page 550),
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        """
        super(ConstraintBasedEstimator, self).__init__(data, **kwargs)


    def test_conditional_independence(self, X, Y, Zs=[]):
        """Chi-square conditional independence test.
        Tests if X is independent from Y given Zs in the data.

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
            This is the seperating set that (potentially) makes X and Y independent.
            Default: []

        Returns
        -------
        p_value: float
            A significance level for the hypothesis that X and Y are dependent
            given Zs. The p_value is the probability of falsely rejecting the
            hypothesis that the variables are conditionally dependent. A low
            p_value (e.g. below 0.05 or 0.01) indicates dependence. (The lower
            the threshold for the p_value, the more likely we are to reject
            dependency, resulting in a sparser graph.)

        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.2.2.3 (page 789)
        [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
        >>> data['E'] = data['A'] + data['B'] + data['C']
        >>> c = ConstraintBasedEstimator(data)
        >>> print(c.test_conditional_independence('A', 'C'))  # independent
        0.9848481578
        >>> print(c.test_conditional_independence('A', 'B', 'D'))  # independent
        0.962206185665
        >>> print(c.test_conditional_independence('A', 'B', ['D', 'E']))  # dependent
        0.0
        """

        if isinstance(Zs, (frozenset, list, set, tuple,)):
            Zs = list(Zs)
        else:
            Zs = [Zs]

        # Check is sample size is sufficient. Require at least 5 samples per parameter (on average)
        # (As suggested in Spirtes et al., Causation, Prediction and Search, 2000, and also used in
        # Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4)
        num_params = ((len(self.state_names[X])-1) *
                      (len(self.state_names[Y])-1) *
                      np.prod([len(self.state_names[Z]) for Z in Zs]))
        if len(self.data) < num_params:
            warn("Insufficient data for testing {0} _|_ {1} | {2}. ".format(X, Y, Zs) +
                 "At least {0} samples recommended, {1} present.".format(num_params, len(self.data)))

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
                                                      Z_state_counts)

        observed = XYZ_state_counts.values.flatten()
        expected = XYZ_expected.values.flatten()
        # remove elements where the expected value is 0;
        # this also corrects the degrees of freedom for chisquare
        observed, expected = zip(*((o, e) for o, e in zip(observed, expected) if not e == 0))

        chi2, p_value = chisquare(observed, expected)

        return p_value
