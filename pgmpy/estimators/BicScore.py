#!/usr/bin/env python
import numpy as np
import pandas as pd
from math import lgamma, log
from pgmpy.estimators import StructureScore


class BicScore(StructureScore):
    def __init__(self, data, **kwargs):
        """
        Class for Bayesian structure scoring for BayesianModels with Dirichlet priors.
        The BIC/MDL score ("Bayesian Information Criterion", also "Minimal Descriptive Length") is a
        log-likelihood score with an additional penalty for network complexity, to avoid overfitting.
        The `score`-method measures how well a model is able to describe the given data set.

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
        super(BicScore, self).__init__(data, **kwargs)

    def score(self, model):
        """
        Computes a score to measure how well the given `BayesianModel` fits to the data set.

        Parameters
        ----------
        model: `BayesianModel` instance
            The Bayesian network that is to be scored. Nodes of the BayesianModel need to coincide
            with column names of data set.

        Returns
        -------
        score: float
            A number indicating the degree of fit between data and model
        """

        score = 0
        for node in model.nodes():
            score += self.local_score(node, model.predecessors(node))
        score += self.structure_prior(model)
        return score

    def structure_prior(self, model):
        "A prior distribution over models. Currently unused (= uniform)."
        return 0

    def local_score(self, variable, parents):
        "Computes a score that roughly measures how much a \
        given variable is influenced by a given list of potential parents."

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        state_counts = self.state_counts(variable, parents)
        sample_size = len(self.data.ix[0])
        num_parents_states = float(len(state_counts))

        score = 0
        for parents_state in state_counts:  # iterate over df columns (only 1 if no parents)
            conditional_sample_size = sum(state_counts[parents_state])

            for state in var_states:
                if state_counts[parents_state][state] > 0:
                    score += state_counts[parents_state][state] * (log(state_counts[parents_state][state]) -
                                                                   log(conditional_sample_size))

        score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)

        return score
