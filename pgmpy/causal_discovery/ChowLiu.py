#!/usr/bin/env python

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.causal_discovery._base import _BaseCausalDiscovery, _TreeSearchMixin


class ChowLiu(_TreeSearchMixin, _BaseCausalDiscovery):
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

    edge_weights_fn : str or callable, default="mutual_info"
        Method to use for computing pairwise edge weights. Options are:

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
    .. [1] Chow, C. K.; Liu, C.N. (1968), "Approximating discrete probability
       distributions with dependence trees", IEEE Transactions on Information
       Theory, IT-14 (3): 462–467

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
    ... )

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
    """

    def __init__(
        self,
        root_node=None,
        edge_weights_fn: str = "mutual_info",
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
        weights = self._get_weights(X, self.edge_weights_fn, self.n_jobs, self.show_progress)

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
