#!/usr/bin/env python

from pgmpy.estimators import BaseEstimator


class StructureScore(BaseEstimator):
    def __init__(self, data, **kwargs):
        """
        Abstract base class for structure scoring classes in pgmpy. Use any of the derived classes
        K2Score, BdeuScore, or BicScore. Scoring classes are
        used to measure how well a model is able to describe the given data set.

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

        Reference
        ---------
        Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.3
        """
        super(StructureScore, self).__init__(data, **kwargs)

    def score(self, model):
        """
        Computes a score to measure how well the given `BayesianModel` fits to the data set.
        (This method relies on the `local_score`-method that is implemented in each subclass.)

        Parameters
        ----------
        model: `BayesianModel` instance
            The Bayesian network that is to be scored. Nodes of the BayesianModel need to coincide
            with column names of data set.

        Returns
        -------
        score: float
            A number indicating the degree of fit between data and model

        Examples
        -------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import K2Score
        >>> # create random data sample with 3 variables, where B and C are identical:
        >>> data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 2)), columns=list('AB'))
        >>> data['C'] = data['B']
        >>> K2Score(data).score(BayesianModel([['A','B'], ['A','C']]))
        -24242.367348745247
        >>> K2Score(data).score(BayesianModel([['A','B'], ['B','C']]))
        -16273.793897051042
        """

        score = 0
        for node in model.nodes():
            score += self.local_score(node, model.predecessors(node))
        score += self.structure_prior(model)
        return score

    def structure_prior(self, model):
        "A (log) prior distribution over models. Currently unused (= uniform)."
        return 0
