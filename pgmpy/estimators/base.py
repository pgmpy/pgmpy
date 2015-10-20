#!/usr/bin/env python


class BaseEstimator(object):
    """
    Base class for estimator class in pgmpy. Estimator class is used for parameter estimation as well
    as structure estimation

    Parameters
    ----------
    model: pgmpy.models.BayesianModel or pgmpy.models.MarkovModel or pgmpy.models.NoisyOrModel
        model for which parameter estimation is to be done

    data: pandas DataFrame object
        datafame object with column names same as the variable names of the network
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data

        get_node_card = lambda _node, _data: _data.ix[:, _node].value_counts().shape[0]
        self.node_card = {_node: get_node_card(_node, data) for _node in self.model.nodes()}
