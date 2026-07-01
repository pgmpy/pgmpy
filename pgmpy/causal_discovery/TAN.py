#!/usr/bin/env python

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.causal_discovery._base import BaseCausalDiscovery, _TreeSearchMixin


class TAN(_TreeSearchMixin, BaseCausalDiscovery):
    """
    Tree-Augmented Naive Bayes (TAN) algorithm for learning a tree-structured
    classifier network.

    TAN extends the Naive Bayes classifier by allowing a maximum-weight spanning
    tree over the feature variables (using conditional mutual information
    I(X; Y | class_node) as edge weights), with additional edges from
    ``class_node`` to every feature variable.

    Parameters
    ----------
    class_node : str, int, or any hashable python object
        The class variable. Edges are added from ``class_node`` to every other
        variable in the final DAG. Must be present in the data passed to
        :meth:`fit`.

    root_node : str, int, or any hashable python object, default=None
        The root node of the feature-variable spanning tree. If ``None``, the
        feature node with the highest sum of marginal edge weights is chosen
        automatically. Must not equal ``class_node``.

    edge_weights_fn : str or callable, default="mutual_info"
        Base scoring function used when computing conditional edge weights
        I(X; Y | class_node). Options are:

        - ``"mutual_info"``: Mutual Information Score.
        - ``"adjusted_mutual_info"``: Adjusted Mutual Information Score.
        - ``"normalized_mutual_info"``: Normalized Mutual Information Score.
        - A callable of the form ``fn(array, array) -> float``.

    n_jobs : int, default=-1
        Number of jobs to run in parallel. ``-1`` means use all processors.

    show_progress : bool, default=True
        If ``True``, shows a progress bar while computing edge weights.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph as a DAG.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of ``causal_graph_``.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    References
    ----------
    .. [1] Friedman N, Geiger D and Goldszmidt M (1997). Bayesian network
       classifiers. Machine Learning 29: 131–163

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import networkx as nx
    >>> import matplotlib.pyplot as plt
    >>> from pgmpy.causal_discovery import TAN
    >>> values = pd.DataFrame(
    ...     np.random.randint(low=0, high=2, size=(1000, 5)),
    ...     columns=["A", "B", "C", "D", "E"],
    ... )

    With a fixed root node and class node:

    >>> est = TAN(class_node="A", root_node="B")
    >>> est.fit(values)
    TAN(class_node='A', root_node='B')
    >>> est.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> nx.draw_circular(
    ...     est.causal_graph_, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
    ... )
    >>> plt.show()

    With automatic root selection:

    >>> est = TAN(class_node="A")
    >>> est.fit(values)
    TAN(class_node='A')
    >>> est.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> nx.draw_circular(
    ...     est.causal_graph_, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
    ... )
    >>> plt.show()
    """

    def __init__(
        self,
        class_node,
        root_node=None,
        edge_weights_fn: str = "mutual_info",
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
        self.class_node = class_node
        self.root_node = root_node
        self.edge_weights_fn = edge_weights_fn
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def _fit(self, X: pd.DataFrame):
        """
        Estimate the ``DAG`` structure that fits best to the given data set
        without parametrization.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from. Each column
            represents one variable.

        Returns
        -------
        self : TAN
            Returns the instance with the fitted attributes set.
        """
        # Step 1: Validate class_node and root_node.
        if self.class_node is None:
            raise ValueError("class_node argument must be specified")
        if self.class_node not in X.columns:
            raise ValueError(f"Class node: {self.class_node} not found in data columns")
        if self.root_node is not None and self.root_node not in X.columns:
            raise ValueError(f"Root node: {self.root_node} not found in data columns.")

        root_node = self.root_node
        # Step 2: Determine root node.
        if root_node is None:
            # Exclude class node
            features = X.drop(columns=[self.class_node])
            weights = self._get_weights(features, self.edge_weights_fn, self.n_jobs, self.show_progress)
            sum_weights = weights.sum(axis=0)
            maxw_idx = np.argsort(sum_weights)[::-1]
            root_node = features.columns[maxw_idx[0]]

        # Step 3: Compute conditional edge weights I(X; Y | class_node).
        weights = self._get_conditional_weights(
            X, self.class_node, self.edge_weights_fn, self.n_jobs, self.show_progress
        )

        # Step 4: Ensure root_node and class_node are distinct.
        if root_node == self.class_node:
            raise ValueError(f"Root node: {root_node} and class node: {self.class_node} are identical")

        # Step 5: Construct Chow-Liu DAG on {data.columns - class_node}.
        class_node_idx = np.where(X.columns == self.class_node)[0][0]
        weights = np.delete(weights, class_node_idx, axis=0)
        weights = np.delete(weights, class_node_idx, axis=1)
        reduced_columns = np.delete(X.columns, class_node_idx)
        D = self._create_tree_and_dag(weights, reduced_columns, root_node)

        # Step 6: Add edges from class_node to every feature node and store.
        D.add_edges_from([(self.class_node, node) for node in reduced_columns])
        self.causal_graph_ = D
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, weight=1, dtype="int")

        return self

    @staticmethod
    def _get_conditional_weights(data, class_node, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True):
        """
        Compute the conditional pairwise edge weight matrix I(X; Y | class_node).

        Each weight is the conditional mutual information between two feature
        variables given the class node, computed as a weighted average over
        the class node's values.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe where each column represents one variable.

        class_node : str
            The class variable to condition on.

        edge_weights_fn : str or callable, default="mutual_info"
            Base scoring function. Options are:

            - ``"mutual_info"``: Mutual Information Score.
            - ``"adjusted_mutual_info"``: Adjusted Mutual Information Score.
            - ``"normalized_mutual_info"``: Normalized Mutual Information Score.
            - A callable of the form ``fn(array, array) -> float``.

        n_jobs : int, default=-1
            Number of jobs to run in parallel. ``-1`` means use all processors.

        show_progress : bool, default=True
            If ``True``, shows a progress bar.

        Returns
        -------
        weights : np.ndarray, shape (n_columns, n_columns)
            Symmetric matrix where ``weights[i, j]`` is I(i; j | class_node).

        """
        # Step 0: Resolve the edge weight computation function.
        edge_weights_fn = _TreeSearchMixin._resolve_edge_weights_fn(edge_weights_fn)

        # Step 1: Compute conditional edge weights for a fully connected graph.
        n_vars = len(data.columns)
        pbar = combinations(data.columns, 2)
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(pbar, total=(n_vars * (n_vars - 1) / 2), desc="Building tree")

        def _conditional_edge_weights_fn(u, v):
            """
            Computes I(u; v | class_node) as a weighted sum over class values.
            """
            cond_marginal = data.loc[:, class_node].value_counts() / data.shape[0]
            cond_edge_weight = 0.0
            for index, marg_prob in cond_marginal.items():
                df_cond_subset = data[data.loc[:, class_node] == index]
                cond_edge_weight += marg_prob * edge_weights_fn(df_cond_subset.loc[:, u], df_cond_subset.loc[:, v])
            return cond_edge_weight

        vals = Parallel(n_jobs=n_jobs)(delayed(_conditional_edge_weights_fn)(u, v) for u, v in pbar)
        weights = np.zeros((n_vars, n_vars))
        indices = np.triu_indices(n_vars, k=1)
        weights[indices] = vals
        weights.T[indices] = vals

        return weights
