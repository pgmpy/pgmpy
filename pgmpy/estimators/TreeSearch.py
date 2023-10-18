#!/usr/bin/env python

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator


class TreeSearch(StructureEstimator):
    """
    Search class for learning tree related graph structure. The algorithms
    supported are Chow-Liu and Tree-augmented naive bayes (TAN).

    Chow-Liu constructs the maximum-weight spanning tree with mutual information
    score as edge weights.

    TAN is an extension of Naive Bayes classifier to allow a tree structure over
    the independent variables to account for interaction.

    Parameters
    ----------
    data: pandas.DataFrame object
        dataframe object where each column represents one variable.

    root_node: str, int, or any hashable python object, default is None.
        The root node of the tree structure. If None then root node is auto-picked
        as the node with the highest sum of edge weights.

    n_jobs: int (default: -1)
        Number of jobs to run in parallel. `-1` means use all processors.

    References
    ----------
    [1] Chow, C. K.; Liu, C.N. (1968), "Approximating discrete probability
        distributions with dependence trees", IEEE Transactions on Information
        Theory, IT-14 (3): 462–467

    [2] Friedman N, Geiger D and Goldszmidt M (1997). Bayesian network classifiers.
        Machine Learning 29: 131–163
    """

    def __init__(self, data, root_node=None, n_jobs=-1, **kwargs):
        if root_node is not None and root_node not in data.columns:
            raise ValueError(f"Root node: {root_node} not found in data columns.")

        self.data = data
        self.root_node = root_node
        self.n_jobs = n_jobs

        super(TreeSearch, self).__init__(data, **kwargs)

    def estimate(
        self,
        estimator_type="chow-liu",
        class_node=None,
        edge_weights_fn="mutual_info",
        show_progress=True,
    ):
        """
        Estimate the `DAG` structure that fits best to the given data set without
        parametrization.

        Parameters
        ----------
        estimator_type: str (chow-liu | tan)
            The algorithm to use for estimating the DAG.

        class_node: string, int or any hashable python object. (optional)
            Needed only if estimator_type = 'tan'. In the estimated DAG, there would be
            edges from class_node to each of the feature variables.

        edge_weights_fn: str or function (default: mutual info)
            Method to use for computing edge weights. By default, Mutual Info Score is
            used.

        show_progress: boolean
            If True, shows a progress bar for the running algorithm.

        Returns
        -------
        Estimated Model: pgmpy.base.DAG
            The estimated model structure.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from pgmpy.estimators import TreeSearch
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> est = TreeSearch(values, root_node='B')
        >>> model = est.estimate(estimator_type='chow-liu')
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy',
        ...                  alpha=0.3)
        >>> plt.show()
        >>> est = TreeSearch(values)
        >>> model = est.estimate(estimator_type='chow-liu')
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy',
        ...                  alpha=0.3)
        >>> plt.show()
        >>> est = TreeSearch(values, root_node='B')
        >>> model = est.estimate(estimator_type='tan', class_node='A')
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy',
        ...                  alpha=0.3)
        >>> plt.show()
        >>> est = TreeSearch(values)
        >>> model = est.estimate(estimator_type='tan')
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy',
        ...                  alpha=0.3)
        >>> plt.show()
        """
        # Step 1. Argument checks
        # Step 1.1: Only chow-liu and tan allowed as estimator type.
        if estimator_type not in {"chow-liu", "tan"}:
            raise ValueError(
                f"Invalid estimator_type. Expected either chow-liu or tan. Got: {estimator_type}"
            )

        # Step 1.2: If estimator_type=tan, class_node must be specified
        if estimator_type == "tan" and class_node is None:
            raise ValueError(
                f"class_node argument must be specified for estimator_type='tan'"
            )
        if estimator_type == "tan" and class_node not in self.data.columns:
            raise ValueError(f"Class node: {class_node} not found in data columns")

        # Step 1.3: If root_node isn't specified, get the node with the highest score.
        weights_computed = False
        if self.root_node is None:
            weights = TreeSearch._get_weights(
                self.data, edge_weights_fn, self.n_jobs, show_progress
            )
            weights_computed = True
            sum_weights = weights.sum(axis=0)
            maxw_idx = np.argsort(sum_weights)[::-1]
            self.root_node = self.data.columns[maxw_idx[0]]

        # Step 2. Compute all edge weights.
        if estimator_type == "chow-liu":
            if not weights_computed:
                weights = TreeSearch._get_weights(
                    self.data, edge_weights_fn, self.n_jobs, show_progress
                )
        else:
            weights = TreeSearch._get_conditional_weights(
                self.data, class_node, edge_weights_fn, self.n_jobs, show_progress
            )

        # Step 3: If estimator_type = "chow-liu", estimate the DAG and return.
        if estimator_type == "chow-liu":
            return TreeSearch._create_tree_and_dag(
                weights, self.data.columns, self.root_node
            )

        # Step 4: If estimator_type = "tan":
        elif estimator_type == "tan":
            # Step 4.1: Checks root_node != class_node
            if self.root_node == class_node:
                raise ValueError(
                    f"Root node: {self.root_node} and class node: {class_node} are identical"
                )

            # Step 4.2: Construct chow-liu DAG on {data.columns - class_node}
            class_node_idx = np.where(self.data.columns == class_node)[0][0]
            weights = np.delete(weights, class_node_idx, axis=0)
            weights = np.delete(weights, class_node_idx, axis=1)
            reduced_columns = np.delete(self.data.columns, class_node_idx)
            D = TreeSearch._create_tree_and_dag(
                weights, reduced_columns, self.root_node
            )

            # Step 4.3: Add edges from class_node to all other nodes.
            D.add_edges_from([(class_node, node) for node in reduced_columns])
            return D

    @staticmethod
    def _get_weights(
        data, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True
    ):
        """
        Helper function to Chow-Liu algorithm for estimating tree structure from given data. Refer to
        pgmpy.estimators.TreeSearch for more details. This function returns the edge weights matrix.

        Parameters
        ----------
        data: pandas.DataFrame object
            dataframe object where each column represents one variable.

        edge_weights_fn: str or function (default: mutual_info)
            Method to use for computing edge weights. Options are:
                1. 'mutual_info': Mutual Information Score.
                2. 'adjusted_mutual_info': Adjusted Mutual Information Score.
                3. 'normalized_mutual_info': Normalized Mutual Information Score.
                4. function(array[n_samples,], array[n_samples,]): Custom function.

        n_jobs: int (default: -1)
            Number of jobs to run in parallel. `-1` means use all processors.

        show_progress: boolean
            If True, shows a progress bar for the running algorithm.

        Returns
        -------
        weights: numpy 2D array, shape = (n_columns, n_columns)
            symmetric matrix where each element represents an edge weight.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.estimators import TreeSearch
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> est = TreeSearch(values, root_node='B')
        >>> model = est.estimate(estimator_type='chow-liu')
        """
        # Step 0: Check for edge weight computation method
        if edge_weights_fn == "mutual_info":
            edge_weights_fn = mutual_info_score
        elif edge_weights_fn == "adjusted_mutual_info":
            edge_weights_fn = adjusted_mutual_info_score
        elif edge_weights_fn == "normalized_mutual_info":
            edge_weights_fn = normalized_mutual_info_score
        elif not callable(edge_weights_fn):
            raise ValueError(
                f"edge_weights_fn should either be 'mutual_info', 'adjusted_mutual_info', "
                f"'normalized_mutual_info'or a function of form fun(array, array). Got: f{edge_weights_fn}"
            )

        # Step 1: Compute edge weights for a fully connected graph.
        n_vars = len(data.columns)
        pbar = combinations(data.columns, 2)
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(pbar, total=(n_vars * (n_vars - 1) / 2), desc="Building tree")

        vals = Parallel(n_jobs=n_jobs)(
            delayed(edge_weights_fn)(data.loc[:, u], data.loc[:, v]) for u, v in pbar
        )
        weights = np.zeros((n_vars, n_vars))
        indices = np.triu_indices(n_vars, k=1)
        weights[indices] = vals
        weights.T[indices] = vals

        return weights

    @staticmethod
    def _get_conditional_weights(
        data, class_node, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True
    ):
        """
        Helper function to TAN (Tree Augmented Naive Bayes) algorithm for
        estimating tree structure from given data. Refer to
        pgmpy.estimators.TreeSearch for more details. This function returns the
        edge weights matrix.

        Parameters
        ----------
        data: pandas.DataFrame object
            dataframe object where each column represents one variable.

        class_node: str
            The class node for TAN. The edge weight is computed as I(X, Y | class_node).

        edge_weights_fn: str or function (default: mutual_info)
            Method to use for computing edge weights. Options are:
                1. 'mutual_info': Mutual Information Score.
                2. 'adjusted_mutual_info': Adjusted Mutual Information Score.
                3. 'normalized_mutual_info': Normalized Mutual Information Score.
                4. function(array[n_samples,], array[n_samples,]): Custom function.

        n_jobs: int (default: -1)
            Number of jobs to run in parallel. `-1` means use all processors.

        show_progress: boolean
            If True, shows a progress bar for the running algorithm.

        Returns
        -------
        weights: numpy 2D array, shape = (n_columns, n_columns)
            symmetric matrix where each element represents an edge weight.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.estimators import TreeSearch
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> est = TreeSearch(values, root_node='B')
        >>> model = est.estimate(estimator_type='tan')
        """
        # Step 0: Check for edge weight computation method
        if edge_weights_fn == "mutual_info":
            edge_weights_fn = mutual_info_score
        elif edge_weights_fn == "adjusted_mutual_info":
            edge_weights_fn = adjusted_mutual_info_score
        elif edge_weights_fn == "normalized_mutual_info":
            edge_weights_fn = normalized_mutual_info_score
        elif not callable(edge_weights_fn):
            raise ValueError(
                f"edge_weights_fn should either be 'mutual_info', 'adjusted_mutual_info', "
                f"'normalized_mutual_info'or a function of form fun(array, array). Got: f{edge_weights_fn}"
            )

        # Step 1: Compute edge weights for a fully connected graph.
        n_vars = len(data.columns)
        pbar = combinations(data.columns, 2)
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(pbar, total=(n_vars * (n_vars - 1) / 2), desc="Building tree")

        def _conditional_edge_weights_fn(u, v):
            """
            Computes the conditional edge weight of variable index u and v conditioned on class_node
            """
            cond_marginal = data.loc[:, class_node].value_counts() / data.shape[0]
            cond_edge_weight = 0
            for index, marg_prob in cond_marginal.items():
                df_cond_subset = data[data.loc[:, class_node] == index]
                cond_edge_weight += marg_prob * edge_weights_fn(
                    df_cond_subset.loc[:, u], df_cond_subset.loc[:, v]
                )
            return cond_edge_weight

        vals = Parallel(n_jobs=n_jobs)(
            delayed(_conditional_edge_weights_fn)(u, v) for u, v in pbar
        )
        weights = np.zeros((n_vars, n_vars))
        indices = np.triu_indices(n_vars, k=1)
        weights[indices] = vals
        weights.T[indices] = vals

        return weights

    @staticmethod
    def _create_tree_and_dag(weights, columns, root_node):
        """
        Helper function to Chow-Liu algorithm for estimating tree structure from given data. Refer to
        pgmpy.estimators.TreeSearch for more details. This function returns the DAG based on the edge weights matrix.

        Parameters
        ----------
        weights: numpy 2D array, shape = (n_columns, n_columns)
            symmetric matrix where each element represents an edge weight.

        columns: list or array
            Names of the columns (& rows) of the weights matrix.

        root_node: str, int, or any hashable python object.
            The root node of the tree structure.

        Returns
        -------
        model: pgmpy.base.DAG
            The estimated model structure.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.estimators import TreeSearch
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> est = TreeSearch(values, root_node='B')
        >>> model = est.estimate(estimator_type='chow-liu')
        """
        # Step 2: Compute the maximum spanning tree using the weights.
        T = nx.maximum_spanning_tree(
            nx.from_pandas_adjacency(
                pd.DataFrame(weights, index=columns, columns=columns),
                create_using=nx.Graph,
            )
        )

        # Step 3: Create DAG by directing edges away from root node and return
        D = nx.bfs_tree(T, root_node)
        return DAG(D)
