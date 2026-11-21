#!/usr/bin/env python

from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.causal_discovery._base import BaseCausalDiscovery, _TreeSearchMixin


class ChowLiu(_TreeSearchMixin, BaseCausalDiscovery):
    """
    Chow-Liu algorithm for learning tree-structured Bayesian networks.

    Constructs the maximum-weight spanning tree over all variables using
    pairwise mutual information as edge weights, then orients every edge
    away from a chosen (or automatically selected) root node.

    Parameters
    ----------
    root_node : str, int, or any hashable python object, default=None
        The root node of the directed tree. If ``None``, the node with the
        highest sum of edge weights is chosen automatically.

    edge_weights_fn : str, callable or None, default=None
        Method to use for computing pairwise edge weights. Options are:

        - ``None``: Chosen from the type of the data (see :func:`~pgmpy.utils.get_dataset_type`):
          ``"mutual_info"`` for discrete data and ``"gaussian_mutual_info"`` for continuous
          data.
        - ``"mutual_info"``: Mutual Information Score.
        - ``"adjusted_mutual_info"``: Adjusted Mutual Information Score.
        - ``"normalized_mutual_info"``: Normalized Mutual Information Score.
        - ``"gaussian_mutual_info"``: Mutual information of jointly Gaussian variables,
          :math:`-\\frac{1}{2} \\ln(1 - \\rho^2)`, for continuous data.
        - ``"mutual_info_regression"``: scikit-learn's mutual information estimate for
          continuous data. Passing ``sklearn.feature_selection.mutual_info_regression`` itself also works.
        - A callable of the form ``fn(array, array) -> float``, called with the integer state codes of two variables
          as stored by pandas (``Series.cat.codes``, typically ``int8``) for categorical variables, or with the
          numeric values for continuous variables.

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

    edge_weights_fn_ : callable
        The function used for computing pairwise edge weights, resolved from ``edge_weights_fn``.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    References
    ----------
    - :footcite:t:`chow_liu_1968`

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import networkx as nx
    >>> import matplotlib.pyplot as plt
    >>> from pgmpy.causal_discovery import ChowLiu
    >>> values = pd.DataFrame(
    ...     np.random.randint(low=0, high=2, size=(1000, 5)),
    ...     columns=["A", "B", "C", "D", "E"],
    ... ).astype("category")

    With a fixed root node:

    >>> est = ChowLiu(root_node="B")
    >>> est.fit(values)
    ChowLiu(root_node='B')
    >>> est.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> nx.draw_circular(
    ...     est.causal_graph_, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
    ... )
    >>> plt.show()

    With automatic root selection:

    >>> est = ChowLiu()
    >>> est.fit(values)
    ChowLiu()
    >>> est.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> nx.draw_circular(
    ...     est.causal_graph_, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
    ... )
    >>> plt.show()

    With continuous data, the Gaussian mutual information is used by default:

    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(size=1000)
    >>> y = x + rng.normal(size=1000)
    >>> z = y + rng.normal(size=1000)
    >>> est = ChowLiu(root_node="x").fit(pd.DataFrame({"x": x, "y": y, "z": z}))
    >>> sorted(est.causal_graph_.edges())
    [('x', 'y'), ('y', 'z')]
    """

    _tags = {
        "name": "chow_liu",
        "data_types": ["discrete", "continuous"],
        "identifiable_graph": "cpdag",
        "requires_target": False,
        "capability:multivariate": True,
        "capability:expert_knowledge": [],
        "assumption:causal_sufficiency": True,
        "assumption:acyclicity": True,
        "assumption:faithfulness": False,
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
        "assumption:non_gaussian_noise": False,
        "assumption:low_noise": False,
    }

    def __init__(
        self,
        root_node=None,
        edge_weights_fn: str | Callable | None = None,
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
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
        self : ChowLiu
            Returns the instance with the fitted attributes set.
        """
        # Step 1: Validate root_node if explicitly provided.
        if self.root_node is not None and self.root_node not in X.columns:
            raise ValueError(f"Root node: {self.root_node} not found in data columns.")

        # Step 2: Compute all pairwise edge weights.
        self.edge_weights_fn_ = self._resolve_edge_weights_fn(self.edge_weights_fn, X)
        weights = self._get_weights(X, self.edge_weights_fn_, self.n_jobs, self.show_progress)

        # Step 3: If root_node isn't specified, auto-pick the node with the
        # highest sum of edge weights.
        root_node = self.root_node
        if root_node is None:
            sum_weights = weights.sum(axis=0)
            maxw_idx = np.argsort(sum_weights)[::-1]
            root_node = X.columns[maxw_idx[0]]

        # Step 4: Build the causal graph and store fitted attributes.
        self.causal_graph_ = self._create_tree_and_dag(weights, X.columns, root_node)
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, weight=1, dtype="int")

        return self
