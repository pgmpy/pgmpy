from pgmpy.inference import Inference
from pgmpy.models import BayesianModel
import pandas as pd
import numpy as np
import networkx as nx
import itertools


class BayesianModelInference(Inference):
    """Inference class specific to Bayesian Models
    """
    def __init__(self, model):
        """Class to calculate probability (pmf) values specific to Bayesian Models

        :param model: instance of BayesianModel model on which inference queries will be computed
        """
        if not isinstance(model, BayesianModel):
            raise TypeError("Model expected type: BayesianModel, got type: ", type(model))
        super(BayesianModelInference, self).__init__(model)

        self.topological_order = list(nx.topological_sort(model))

    def pre_compute_reduce(self, variable):
        """Get probability arrays for a node as function of conditional dependencies

        Internal function used for Bayesian networks, eg. in BayesianModelSampling
        and BayesianModelProbability.

        :param variable: node of the Bayesian network
        :return: dict with probability array for node as function of conditional dependency values
        """
        variable_cpd = self.model.get_cpds(variable)
        variable_evid = variable_cpd.variables[:0:-1]
        cached_values = {}

        for state_combination in itertools.product(
                *[range(self.cardinality[var]) for var in variable_evid]
        ):
            states = list(zip(variable_evid, state_combination))
            cached_values[state_combination] = variable_cpd.reduce(
                states, inplace=False
            ).values

        return cached_values


class BayesianModelProbability(BayesianModelInference):
    """Class to calculate probability (pmf) values specific to Bayesian Models
    """
    def __init__(self, model):
        """Class to calculate probability (pmf) values specific to Bayesian Models

        :param model: instance of BayesianModel model on which inference queries will be computed
        """
        super(BayesianModelProbability, self).__init__(model)

    def _log_probability_node(self, X, ordering, node):
        """Evaluate the log probability of each datapoint for a specific node.

        Internal function used by log_probability().

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :param list ordering: ordering of columns in X, used by the Bayesian model.
            default is topological ordering used by model.
        :param node: node from the Bayesian network.
        :return: ndarray, shape (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        def vec_translate(a, my_dict):
            return np.vectorize(my_dict.__getitem__)(a)

        cpd = self.model.get_cpds(node)

        # variable to probe: x[n], where n is the node number
        current = cpd.variables[0]
        current_idx = ordering.index(current)
        current_val = X[:, current_idx]
        current_no = vec_translate(current_val, cpd.name_to_no[current])

        # conditional dependencies E of the probed variable
        evidence = cpd.variables[:0:-1]
        evidence_idx = [ordering.index(ev) for ev in evidence]
        evidence_val = X[:, evidence_idx]
        evidence_no = np.empty_like(evidence_val)
        for i, ev in enumerate(evidence):
            evidence_no[:, i] = vec_translate(evidence_val[:, i], cpd.name_to_no[ev])

        if evidence:
            # there are conditional dependencies E for x[n] for this node
            # Here we retrieve the array: p(x[n]|E). We do this for each x in X.
            # We pick the specific node value from the arrays below.
            cached_values = self.pre_compute_reduce(variable=node)
            weights = np.array([cached_values[tuple(en)] for en in evidence_no])
        else:
            # there are NO conditional dependencies for this node
            # retrieve array: p(x[n]).  We do this for each x in X.
            # We pick the specific node value from the arrays below.
            weights = np.array([cpd.values] * len(X))

        # pick the specific node value x[n] from the array p(x[n]|E) or p(x[n])
        # We do this for each x in X.
        probability_node = np.array([weights[i][cn] for i, cn in enumerate(current_no)])

        return np.log(probability_node)

    def log_probability(self, X, ordering=None):
        """Evaluate the logarithmic probability of each point in a data set.

        :param X: pandas dataframe OR array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :param list ordering: ordering of columns in X, used by the Bayesian model.
            default is topological ordering used by model.
        :return: ndarray, shape (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        if isinstance(X, pd.DataFrame):
            # use numpy array from now on.
            ordering = X.columns.to_list()
            X = X.values
        if ordering is None:
            ordering = self.topological_order

        logp = np.array([self._log_probability_node(X, ordering, node) for node in self.topological_order])
        return np.sum(logp, axis=0)

    def score(self, X, ordering=None):
        """Compute the total log probability density under the model.

        :param X: pandas dataframe OR array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :param list ordering: ordering of columns in X, used by the Bayesian model.
            default is topological ordering used by model.
        :return: float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return np.sum(self.log_probability(X, ordering))
