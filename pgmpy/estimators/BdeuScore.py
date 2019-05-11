#!/usr/bin/env python
from math import lgamma

from pgmpy.estimators import StructureScore


class BdeuScore(StructureScore):
    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        """
        Class for Bayesian structure scoring for BayesianModels with Dirichlet priors.
        The BDeu score is the result of setting all Dirichlet hyperparameters/pseudo_counts to
        `equivalent_sample_size/variable_cardinality`.
        The `score`-method measures how well a model is able to describe the given data set.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        equivalent_sample_size: int (default: 10)
            The equivalent/imaginary sample size (of uniform pseudo samples) for the dirichlet hyperparameters.
            The score is sensitive to this value, runs with different values might be useful.

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
        ---------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.3.4-18.3.6 (esp. page 806)
        [2] AM Carvalho, Scoring functions for learning Bayesian networks,
        http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
        """
        self.equivalent_sample_size = equivalent_sample_size
        super(BdeuScore, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        state_counts = self.state_counts(variable, parents)
        num_parents_states = float(len(state_counts.columns))

        score = 0
        for (
            parents_state
        ) in state_counts:  # iterate over df columns (only 1 if no parents)
            conditional_sample_size = sum(state_counts[parents_state])

            score += lgamma(self.equivalent_sample_size / num_parents_states) - lgamma(
                conditional_sample_size
                + self.equivalent_sample_size / num_parents_states
            )

            for state in var_states:
                if state_counts[parents_state][state] > 0:
                    score += lgamma(
                        state_counts[parents_state][state]
                        + self.equivalent_sample_size
                        / (num_parents_states * var_cardinality)
                    ) - lgamma(
                        self.equivalent_sample_size
                        / (num_parents_states * var_cardinality)
                    )
        return score
