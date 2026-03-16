#!/usr/bin/env python

from itertools import combinations
from typing import Callable, Dict, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator


class TreeSearch(SklearnBaseEstimator, StructureEstimator):
    """
    sklearn-compatible Tree Structure Learning Algorithm.

    Supports Chow-Liu and Tree-augmented Naive Bayes (TAN) algorithms
    for learning tree-structured Bayesian networks from data.

    This class follows sklearn conventions for compatibility with sklearn's
    model selection tools (e.g., GridSearchCV, Pipeline).

    Parameters
    ----------
    estimator_type : str, default="chow-liu"
        The algorithm to use. Options:
        - "chow-liu": Constructs maximum-weight spanning tree
        - "tan": Tree-augmented Naive Bayes

    root_node : str, int, or hashable, optional
        Root node of the tree. If None, auto-selected based on edge weights.

    class_node : str, int, or hashable, optional
        Required for TAN estimator. Class node for conditional probabilities.

    edge_weights_fn : str or callable, default="mutual_info"
        Edge weight computation method:
        - "mutual_info": Mutual Information Score
        - "adjusted_mutual_info": Adjusted Mutual Information Score
        - "normalized_mutual_info": Normalized Mutual Information Score
        - callable: Custom function with signature f(array, array) -> float

    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all processors.

    show_progress : bool, default=False
        Whether to display progress bar during learning.

    Attributes
    ----------
    model_ : pgmpy.base.DAG
        The learned DAG structure (set after calling fit()).

    root_node_ : str, int, or hashable
        The root node used in the learned structure.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import TreeSearch
    >>>
    >>> # Generate sample data
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(1000, 5)), columns=["A", "B", "C", "D", "E"]
    ... )
    >>>
    >>> # Chow-Liu learning
    >>> ts = TreeSearch(estimator_type="chow-liu", root_node="A")
    >>> ts.fit(data)
    >>> dag = ts.model_
    >>>
    >>> # TAN learning
    >>> ts_tan = TreeSearch(estimator_type="tan", root_node="B", class_node="A")
    >>> ts_tan.fit(data)
    >>> dag_tan = ts_tan.model_
    >>>
    >>> # sklearn integration
    >>> from sklearn.model_selection import GridSearchCV
    >>> param_grid = {
    ...     "root_node": ["A", "B", None],
    ...     "edge_weights_fn": ["mutual_info", "adjusted_mutual_info"],
    ... }
    >>> gs = GridSearchCV(TreeSearch(), param_grid)
    >>> gs.fit(data)
    """

    def __init__(
        self,
        estimator_type: str = "chow-liu",
        root_node: Optional[Union[str, int]] = None,
        class_node: Optional[Union[str, int]] = None,
        edge_weights_fn: Union[str, Callable] = "mutual_info",
        n_jobs: int = -1,
        show_progress: bool = False,
    ):
        self.estimator_type = estimator_type
        self.root_node = root_node
        self.class_node = class_node
        self.edge_weights_fn = edge_weights_fn
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def fit(self, X: pd.DataFrame, y=None) -> "TreeSearch":
        """
        Learn the tree structure from data.

        sklearn-compatible fit method that learns the DAG structure
        from the input data matrix.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data where each column represents a variable.
            Shape: (n_samples, n_features)

        y : None
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        self : TreeSearch
            Returns self for method chaining compatibility.

        Raises
        ------
        ValueError
            If estimator_type is invalid or class_node is required but not provided.
        TypeError
            If X is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")

        # Store data and update root_node_ if needed
        self.data = X
        self.root_node_ = self.root_node

        # Validation
        self._validate_parameters()

        # Call the core estimation logic
        self.model_ = self._estimate_structure()

        return self

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self.estimator_type not in {"chow-liu", "tan"}:
            raise ValueError(
                f"Invalid estimator_type. Expected 'chow-liu' or 'tan'. "
                f"Got: {self.estimator_type}"
            )

        if self.estimator_type == "tan" and self.class_node is None:
            raise ValueError("class_node must be specified when estimator_type='tan'")

        if self.class_node is not None and self.class_node not in self.data.columns:
            raise ValueError(
                f"class_node '{self.class_node}' not found in data columns"
            )

        if self.root_node is not None and self.root_node not in self.data.columns:
            raise ValueError(f"root_node '{self.root_node}' not found in data columns")

    def _estimate_structure(self) -> DAG:
        """Core structure estimation logic."""
        # Step 1: Select root_node if not specified
        if self.root_node_ is None:
            weights = self._get_weights(
                self.data, self.edge_weights_fn, self.n_jobs, self.show_progress
            )
            sum_weights = weights.sum(axis=0)
            max_idx = np.argsort(sum_weights)[::-1]
            self.root_node_ = self.data.columns[max_idx[0]]

        # Step 2: Compute edge weights
        if self.estimator_type == "chow-liu":
            weights = self._get_weights(
                self.data, self.edge_weights_fn, self.n_jobs, self.show_progress
            )
            return self._create_tree_and_dag(
                weights, self.data.columns, self.root_node_
            )

        elif self.estimator_type == "tan":
            if self.root_node_ == self.class_node:
                raise ValueError(
                    f"root_node and class_node cannot be identical. "
                    f"Got both as '{self.root_node_}'"
                )

            weights = self._get_conditional_weights(
                self.data,
                self.class_node,
                self.edge_weights_fn,
                self.n_jobs,
                self.show_progress,
            )

            # Remove class_node from weights matrix
            class_idx = np.where(self.data.columns == self.class_node)[0][0]
            weights = np.delete(weights, class_idx, axis=0)
            weights = np.delete(weights, class_idx, axis=1)
            reduced_columns = np.delete(self.data.columns, class_idx)

            # Create tree on reduced columns
            dag = self._create_tree_and_dag(weights, reduced_columns, self.root_node_)

            # Add edges from class_node to all other nodes
            dag.add_edges_from([(self.class_node, node) for node in reduced_columns])
            return dag

    @staticmethod
    def _get_weights(
        data: pd.DataFrame,
        edge_weights_fn: Union[str, Callable] = "mutual_info",
        n_jobs: int = -1,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Compute edge weights for fully connected graph (Chow-Liu).

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        edge_weights_fn : str or callable
            Edge weight computation method
        n_jobs : int
            Number of parallel jobs
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        weights : np.ndarray
            Symmetric weight matrix of shape (n_vars, n_vars)
        """
        # Resolve weight function
        weight_func = TreeSearch._resolve_weight_function(edge_weights_fn)

        n_vars = len(data.columns)
        pbar = combinations(data.columns, 2)

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(
                pbar,
                total=(n_vars * (n_vars - 1) / 2),
                desc="Computing edge weights",
            )

        # Compute weights in parallel
        vals = Parallel(n_jobs=n_jobs)(
            delayed(weight_func)(data.loc[:, u], data.loc[:, v]) for u, v in pbar
        )

        # Build symmetric weight matrix
        weights = np.zeros((n_vars, n_vars))
        indices = np.triu_indices(n_vars, k=1)
        weights[indices] = vals
        weights.T[indices] = vals

        return weights

    @staticmethod
    def _get_conditional_weights(
        data: pd.DataFrame,
        class_node: Union[str, int],
        edge_weights_fn: Union[str, Callable] = "mutual_info",
        n_jobs: int = -1,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Compute conditional edge weights for TAN algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        class_node : str or int
            Class node for conditioning
        edge_weights_fn : str or callable
            Edge weight computation method
        n_jobs : int
            Number of parallel jobs
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        weights : np.ndarray
            Symmetric weight matrix of shape (n_vars, n_vars)
        """
        weight_func = TreeSearch._resolve_weight_function(edge_weights_fn)

        n_vars = len(data.columns)
        pbar = combinations(data.columns, 2)

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(
                pbar,
                total=(n_vars * (n_vars - 1) / 2),
                desc="Computing conditional edge weights",
            )

        def _conditional_weight(u, v):
            """Compute conditional edge weight I(u, v | class_node)."""
            cond_marginal = data.loc[:, class_node].value_counts() / data.shape[0]
            cond_weight = 0.0

            for class_value, prob in cond_marginal.items():
                subset = data[data.loc[:, class_node] == class_value]
                cond_weight += prob * weight_func(subset.loc[:, u], subset.loc[:, v])

            return cond_weight

        vals = Parallel(n_jobs=n_jobs)(
            delayed(_conditional_weight)(u, v) for u, v in pbar
        )

        weights = np.zeros((n_vars, n_vars))
        indices = np.triu_indices(n_vars, k=1)
        weights[indices] = vals
        weights.T[indices] = vals

        return weights

    @staticmethod
    def _create_tree_and_dag(
        weights: np.ndarray,
        columns,
        root_node: Union[str, int],
    ) -> DAG:
        """
        Create DAG from maximum spanning tree.

        Parameters
        ----------
        weights : np.ndarray
            Weight matrix of shape (n_vars, n_vars)
        columns : list or array
            Column names
        root_node : str or int
            Root node for DAG orientation

        Returns
        -------
        dag : pgmpy.base.DAG
            The learned DAG
        """
        # Create maximum spanning tree
        graph_df = pd.DataFrame(weights, index=columns, columns=columns)
        undirected_tree = nx.maximum_spanning_tree(
            nx.from_pandas_adjacency(graph_df, create_using=nx.Graph)
        )

        # Direct tree from root using BFS
        directed_tree = nx.bfs_tree(undirected_tree, root_node)
        return DAG(directed_tree)

    @staticmethod
    def _resolve_weight_function(
        edge_weights_fn: Union[str, Callable],
    ) -> Callable:
        """
        Resolve edge weight function name to callable.

        Parameters
        ----------
        edge_weights_fn : str or callable
            Weight function name or callable

        Returns
        -------
        callable
            The weight function

        Raises
        ------
        ValueError
            If function name is not recognized
        """
        if callable(edge_weights_fn):
            return edge_weights_fn

        weight_functions = {
            "mutual_info": mutual_info_score,
            "adjusted_mutual_info": adjusted_mutual_info_score,
            "normalized_mutual_info": normalized_mutual_info_score,
        }

        if edge_weights_fn not in weight_functions:
            raise ValueError(
                f"Invalid edge_weights_fn '{edge_weights_fn}'. "
                f"Expected one of {list(weight_functions.keys())} or a callable."
            )

        return weight_functions[edge_weights_fn]

    def get_params(self, deep: bool = True) -> Dict[str, any]:
        """
        Get parameters for this estimator.

        sklearn-compatible method for hyperparameter retrieval.
        Required for sklearn's GridSearchCV and Pipeline.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            "estimator_type": self.estimator_type,
            "root_node": self.root_node,
            "class_node": self.class_node,
            "edge_weights_fn": self.edge_weights_fn,
            "n_jobs": self.n_jobs,
            "show_progress": self.show_progress,
        }
        return params

    def set_params(self, **params) -> "TreeSearch":
        """
        Set the parameters of this estimator.

        sklearn-compatible method for hyperparameter setting.
        Required for sklearn's GridSearchCV and Pipeline.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : TreeSearch
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If invalid parameters are provided.
        """
        if not params:
            return self

        valid_params = set(self.get_params().keys())
        invalid_params = set(params.keys()) - valid_params

        if invalid_params:
            raise ValueError(
                f"Invalid parameters: {invalid_params}. "
                f"Valid parameters are: {valid_params}"
            )

        for key, value in params.items():
            setattr(self, key, value)

        return self
