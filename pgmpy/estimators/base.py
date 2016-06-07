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

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.
    """
    def __init__(self, model, data, state_names=None):
        self.model = model
        self.data = data
        if not isinstance(state_names, dict):
            self.state_names = {node: self._get_state_names(node) for node in model.nodes()}
        else:
            self.state_names = dict()
            for node in model.nodes():
                if node in state_names:
                    if not set(self._get_state_names(node)) <= set(state_names[node]):
                        raise ValueError("Data contains unexpected states for variable '{0}'.".format(str(node)))
                    self.state_names[node] = sorted(state_names[node])
                else:
                    self.state_names[node] = self._get_state_names(node)

    def _get_state_names(self, variable):
        states = sorted(list(self.data.ix[:, variable].unique()))
        return states
