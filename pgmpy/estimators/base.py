#!/usr/bin/env python
import numpy as np


class BaseEstimator(object):
    """
    Base class for parameter estimators in pgmpy.

    Parameters
    ----------
    model: pgmpy.models.BayesianModel or pgmpy.models.MarkovModel or pgmpy.models.NoisyOrModel
        model for which parameter estimation is to be done

    data: pandas DataFrame object
        datafame object with column names identical to the variable names of the model

    node_values: dict (optional)
        A dict indicating, for each variable, the discrete set of values (realizations)
        that the variable can take. If unspecified, the observed values in the data set
        are taken as the only possible states.
    """
    def __init__(self, model, data, node_values=None):
        self.model = model
        self.data = data.astype(np.int)
        if not isinstance(node_values, dict):
            self.node_values = {node: self._get_node_values(node) for node in model.nodes()}
        else:
            self.node_values = dict()
            for node in model.nodes():
                if node in node_values:
                    if not set(self._get_node_values(node)) <= set(node_values[node]):
                        raise ValueError("Data contains unexpected values for variable '" + str(node) + "'.")
                    self.node_values[node] = node_values[node]
                else:
                    self.node_values[node] = self._get_node_values(node)

    def _get_node_values(self, node):
        values = list(self.data.ix[:, node].unique())
        return values
