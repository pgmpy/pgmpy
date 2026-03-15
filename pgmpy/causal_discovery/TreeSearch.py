from typing import Callable, Hashable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.estimators._tree_search_utils import (
    _create_tree_and_dag,
    _get_conditional_weights,
    _get_weights,
)


class TreeSearch(_BaseCausalDiscovery):
    """
    Tree-structured causal discovery using Chow-Liu and TAN.

    The algorithm supports:
    - Chow-Liu: maximum-weight spanning tree using mutual information.
    - TAN: tree-augmented Naive Bayes with a class node.

    Parameters
    ----------
    estimator_type : str, default="chow-liu"
        The algorithm to use for estimating the DAG. Supported values are
        "chow-liu" and "tan".

    class_node : hashable, default=None
        The class node for TAN. Required when estimator_type="tan".

    edge_weights_fn : str or callable, default="mutual_info"
        Method to use for computing edge weights. Supported strings are:
        "mutual_info", "adjusted_mutual_info", "normalized_mutual_info".
        Custom callables should take two 1D arrays and return a scalar.

    root_node : hashable, default=None
        Root node for orienting the tree edges. If None, the root is chosen
        as the node with the highest sum of edge weights.

    n_jobs : int, default=-1
        Number of jobs to run in parallel. `-1` means use all processors.

    show_progress : bool, default=True
        If True, shows a progress bar for the running algorithm.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned tree-structured DAG.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    root_node_ : hashable
        The root node used to orient the tree edges during fitting.

    class_node_ : hashable or None
        The class node used for TAN, if applicable.
    """

    def __init__(
        self,
        estimator_type: str = "chow-liu",
        class_node: Optional[Hashable] = None,
        edge_weights_fn: Union[str, Callable] = "mutual_info",
        root_node: Optional[Hashable] = None,
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
        self.estimator_type = estimator_type
        self.class_node = class_node
        self.edge_weights_fn = edge_weights_fn
        self.root_node = root_node
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def _fit(self, X: pd.DataFrame):
        self.variables_ = list(X.columns)

        if self.estimator_type not in {"chow-liu", "tan"}:
            raise ValueError(
                "Invalid estimator_type. Expected either chow-liu or tan. "
                f"Got: {self.estimator_type}"
            )

        if self.root_node is not None and self.root_node not in X.columns:
            raise ValueError(f"Root node: {self.root_node} not found in data columns.")

        if self.estimator_type == "tan":
            if self.class_node is None:
                raise ValueError(
                    "class_node argument must be specified for estimator_type='tan'"
                )
            if self.class_node not in X.columns:
                raise ValueError(
                    f"Class node: {self.class_node} not found in data columns"
                )

        weights_computed = False
        root_node = self.root_node
        if root_node is None:
            weights = _get_weights(
                X,
                edge_weights_fn=self.edge_weights_fn,
                n_jobs=self.n_jobs,
                show_progress=self.show_progress,
            )
            weights_computed = True
            sum_weights = weights.sum(axis=0)
            maxw_idx = np.argsort(sum_weights)[::-1]
            root_node = X.columns[maxw_idx[0]]

        self.root_node_ = root_node
        self.class_node_ = self.class_node

        if self.estimator_type == "chow-liu":
            if not weights_computed:
                weights = _get_weights(
                    X,
                    edge_weights_fn=self.edge_weights_fn,
                    n_jobs=self.n_jobs,
                    show_progress=self.show_progress,
                )
            self.causal_graph_ = _create_tree_and_dag(weights, X.columns, root_node)
        else:
            if root_node == self.class_node:
                raise ValueError(
                    f"Root node: {root_node} and class node: {self.class_node} are identical"
                )

            weights = _get_conditional_weights(
                X,
                self.class_node,
                edge_weights_fn=self.edge_weights_fn,
                n_jobs=self.n_jobs,
                show_progress=self.show_progress,
            )

            class_node_idx = np.where(X.columns == self.class_node)[0][0]
            weights = np.delete(weights, class_node_idx, axis=0)
            weights = np.delete(weights, class_node_idx, axis=1)
            reduced_columns = np.delete(X.columns, class_node_idx)
            self.causal_graph_ = _create_tree_and_dag(
                weights, reduced_columns, root_node
            )
            self.causal_graph_.add_edges_from(
                [(self.class_node, node) for node in reduced_columns]
            )

        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, weight=1, dtype="int"
        )
        return self
