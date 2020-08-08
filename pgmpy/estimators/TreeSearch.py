#!/usr/bin/env python

from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import (
    mutual_info_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)

from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator
from pgmpy.global_vars import SHOW_PROGRESS


class TreeSearch(StructureEstimator):
    def __init__(self, data, root_node, n_jobs=-1, **kwargs):
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

        root_node: str, int, or any hashable python object.
            The root node of the tree structure.

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
        if root_node not in data.columns:
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
            Required if estimator_type = 'tan'.

        edge_weights_fn: str or function (default: mutual info)
            Method to use for computing edge weights. By default Mutual Info Score is
            used.

        show_progress: boolean
            If True, shows a progress bar for the running algorithm.

        Returns
        -------
        model: `pgmpy.base.DAG` instance
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
        >>> est = TreeSearch(values, root_node='B')
        >>> model = est.estimate(estimator_type='tan', class_node='A')
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy',
        ...                  alpha=0.3)
        >>> plt.show()
        """
        # Step 1. Argument checks
        if estimator_type not in {"chow-liu", "tan"}:
            raise ValueError(
                f"Invalid estimator_type. Expected either chow-liu or tan. Got: {self.return_type}"
            )

        # Step 2: If estimator_type = "chow-liu", estimate the DAG and return.
        if estimator_type == "chow-liu":
            return TreeSearch.chow_liu(
                self.data, self.root_node, edge_weights_fn, self.n_jobs, show_progress
            )

        # Step 3: If estimator_type = "tan":
        elif estimator_type == "tan":

            # Step 3.1: Checks for class_node and root_node != class_node
            if class_node is None:
                raise ValueError("class_node must be specified for estimator_type=tiu")

            elif class_node not in self.data.columns:
                raise ValueError(f"Class node: {class_node} not found in data columns")

            elif self.root_node == class_node:
                raise ValueError(
                    f"Root node: {self.root_node} and class node: {class_node} are identical"
                )

            # Step 3.2:: Construct chow-liu on {data.columns - class_node}
            df_features = self.data.loc[:, self.data.columns != class_node]
            D = TreeSearch.chow_liu(
                df_features, self.root_node, edge_weights_fn, self.n_jobs, show_progress
            )

            # Step 3.3: Add edges from class_node to all other nodes.
            D.add_edges_from([(class_node, node) for node in df_features.columns])

            return D

    @staticmethod
    def chow_liu(
        data, root_node, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True
    ):
        """
        Chow-Liu algorithm for estimating tree structure from given data. Refer to
        pgmpy.estimators.TreeSearch for more details.

        Parameters
        ----------
        data: pandas.DataFrame object
            dataframe object where each column represents one variable.

        root_node: str, int, or any hashable python object.
            The root node of the tree structure.

        n_jobs: int (default: -1)
            Number of jobs to run in parallel. `-1` means use all processors.

        edge_weights_fn: str or function (default: mutual_info)
            Method to use for computing edge weights. Options are:
                1. 'mutual_info': Mutual Information Score.
                2. 'adjusted_mutual_info': Adjusted Mutual Information Score.
                3. 'normalized_mutual_info': Normalized Mutual Information Score.
                4. function(array[n_samples,], array[n_samples,]): Custom function.

        show_progress: boolean
            If True, shows a progress bar for the running algorithm.

        Returns
        -------
        model: `pgmpy.base.DAG` instance
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
        # Step 0: Check for edge weight computation method
        if edge_weights_fn == "mutual_info":
            edge_weights_fn = mutual_info_score
        elif not isinstance(edge_weights_fn, callable):
            raise ValueError(
                f"edge_weights_fn should either be 'mutual_info', 'adjusted_mutual_info', "
                f"'normalized_mutual_info'or a function of form fun(array, array). Got: f{edge_weights_fn}"
            )

        # Step 1: Compute edge weights for a fully connected graph.
        n_vars = len(data.columns)
        if show_progress and SHOW_PROGRESS:
            pbar = tqdm(
                combinations(data.columns, 2), total=(n_vars * (n_vars - 1) / 2)
            )
            pbar.set_description("Building tree")
        else:
            pbar = combinations(data.columns, 2)

        vals = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(edge_weights_fn)(data.loc[:, u], data.loc[:, v]) for u, v in pbar
        )
        weights = np.zeros((n_vars, n_vars))
        weights[np.triu_indices(n_vars, k=1)] = vals

        # Step 2: Compute the maximum spanning tree using the weights.
        T = nx.maximum_spanning_tree(
            nx.from_pandas_adjacency(
                pd.DataFrame(weights, index=data.columns, columns=data.columns),
                create_using=nx.Graph,
            )
        )

        # Step 3: Create DAG by directing edges away from root node and return
        D = nx.bfs_tree(T, root_node)
        return DAG(D)
