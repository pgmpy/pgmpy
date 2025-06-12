import numpy as np
import pandas as pd
import warnings
from pgmpy.estimators.LogLikelihoodBase import LogLikelihoodBase
import networkx as nx

from pgmpy.sampling import BayesianModelInference


class BayesianModelProbability(LogLikelihoodBase):
    """
    Class for computing probability of data given a Bayesian network model.

    This class is now a wrapper around the unified LogLikelihoodScore implementation.
    It is kept for backward compatibility.

    Parameters
    ----------
    model : pgmpy.models.BayesianNetwork"""

    def __init__(self, model):
        self.model = model
        self.topological_order = list(nx.topological_sort(model))

    def local_score(self, variable, parents):
        """
        Compute the local score for a variable given its parents.

        Parameters
        ----------
        variable : str
            The variable for which to compute the score.
        parents : list
            List of parent variables.

        Returns
        -------
        float
            The local score for the variable.
        """
        cpd = self.model.get_cpds(variable)
        if not cpd:
            raise ValueError(f"No CPD found for variable {variable}")

        # Get the values from the CPD
        values = cpd.values

        # Compute the log probability
        log_prob = np.log(values)

        # Sum over all possible values
        return np.sum(log_prob)

    def log_probability(self, data, ordering=None):
        """
        Evaluate the logarithmic probability of each point in a data set.

        Parameters
        ----------
        data : pandas.DataFrame or array_like
            List of n_features-dimensional data points.
        ordering : list, optional
            Ordering of columns in data, used by the Bayesian model.
            Default is topological ordering used by model.

        Returns
        -------
        np.array
            The array of log(density) evaluations.
        """
        if isinstance(data, pd.DataFrame):
            ordering = data.columns.to_list()
            data = data.values
        if ordering is None:
            ordering = self.topological_order
            data = data.loc[:, ordering].values

        from pgmpy.estimators import LogLikelihoodScore

        score = LogLikelihoodScore(pd.DataFrame(data, columns=ordering), use_cpd=True)
        return score.score(self.model)

    def score(self, data, ordering=None):
        """
        Compute the total log probability density under the model.

        Parameters
        ----------
        data : pandas.DataFrame or array_like
            List of n_features-dimensional data points.
        ordering : list, optional
            Ordering of columns in data, used by the Bayesian model.
            Default is topological ordering used by model.

        Returns
        -------
        float
            The log-likelihood of data.
        """
        return np.sum(self.log_probability(data, ordering))
