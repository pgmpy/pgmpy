import numpy as np
import pandas as pd

from pgmpy.sampling import BayesianModelInference


class BayesianModelProbability(BayesianModelInference):
    """
    Class to calculate probability (pmf) values specific to Bayesian Models
    """

    def __init__(self, model):
        """
        Class to calculate probability (pmf) values specific to Bayesian Models

        Parameters
        ----------
        model: Bayesian Model
            model on which inference queries will be computed
        """
        super(BayesianModelProbability, self).__init__(model)

    def _log_probability_node(self, data, ordering, node):
        """
        Evaluate the log probability of each datapoint for a specific node.

        Internal function used by log_probability().

        Parameters
        ----------
        data: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        ordering: list
            ordering of columns in data, used by the Bayesian model.
            default is topological ordering used by model.

        node: Bayesian Model Node
            node from the Bayesian network.

        Returns
        -------
        Log probability of node: np.array (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """

        def vec_translate(a, my_dict):
            return np.vectorize(my_dict.__getitem__)(a)

        cpd = self.model.get_cpds(node)

        # variable to probe: data[n], where n is the node number
        current = cpd.variables[0]
        current_idx = ordering.index(current)
        current_val = data[:, current_idx]
        current_no = vec_translate(current_val, cpd.name_to_no[current])

        # conditional dependencies E of the probed variable
        evidence = [var for var in cpd.variables[1:] if var not in self.model.latents]
        evidence_idx = [ordering.index(ev) for ev in evidence]
        evidence_val = data[:, evidence_idx]
        evidence_no = np.empty_like(evidence_val, dtype=int)
        for i, ev in enumerate(evidence):
            evidence_no[:, i] = vec_translate(evidence_val[:, i], cpd.name_to_no[ev])

        if evidence:
            # there are conditional dependencies E for data[n] for this node
            # Here we retrieve the array: p(x[n]|E). We do this for each x in data.
            # We pick the specific node value from the arrays below.

            unique, inverse = np.unique(evidence_no, axis=0, return_inverse=True)
            unique = [tuple(u) for u in unique]
            state_to_index, index_to_weight = self.pre_compute_reduce_maps(
                variable=node, evidence=evidence, state_combinations=unique
            )
            weights = np.array(
                [index_to_weight[state_to_index[tuple(u)]] for u in unique]
            )[inverse]
        else:
            # there are NO conditional dependencies for this node
            # retrieve array: p(x[n]).  We do this for each x in data.
            # We pick the specific node value from the arrays below.
            weights = np.array([cpd.values] * len(data))

        # pick the specific node value x[n] from the array p(x[n]|E) or p(x[n])
        # We do this for each x in data.
        probability_node = np.array([weights[i][cn] for i, cn in enumerate(current_no)])

        return np.log(probability_node)

    def log_probability(self, data, ordering=None):
        """
        Evaluate the logarithmic probability of each point in a data set.

        Parameters
        ----------
        data: pandas dataframe OR array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        ordering: list
            ordering of columns in data, used by the Bayesian model.
            default is topological ordering used by model.

        Returns
        -------
        Log probability of each datapoint: np.array (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        if isinstance(data, pd.DataFrame):
            # use numpy array from now on.
            ordering = data.columns.to_list()
            data = data.values
        if ordering is None:
            ordering = self.topological_order
            data = data.loc[:, ordering].values

        logp = np.array(
            [self._log_probability_node(data, ordering, node) for node in ordering]
        )
        return np.sum(logp, axis=0)

    def score(self, data, ordering=None):
        """
        Compute the total log probability density under the model.

        Parameters
        ----------
        data: pandas dataframe OR array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        ordering: list
            ordering of columns in data, used by the Bayesian model.
            default is topological ordering used by model.

        Returns
        -------
        Log-likelihood of data: float
            This is normalized to be a probability density, so the value
            will be low for high-dimensional data.
        """
        return np.sum(self.log_probability(data, ordering))
