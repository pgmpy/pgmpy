from collections.abc import Hashable
from typing import Optional

import networkx as nx
import pandas as pd

from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.estimators import TreeSearch as TreeSearchEstimator


class TreeSearch(_BaseCausalDiscovery):
    """
    Tree-based causal discovery using Chow-Liu or Tree-Augmented Naive Bayes (TAN).

    This class implements two tree-based causal discovery algorithms:

    1. **Chow-Liu** [1]_: Constructs the maximum-weight spanning tree using
       mutual information as edge weights, then directs edges away from the
       root node to produce a DAG.

    2. **TAN** [2]_: An extension of Naive Bayes that allows a tree structure
       over the feature variables to capture pairwise interactions, conditioned
       on a class node.

    Parameters
    ----------
    estimator_type : str, default='chow-liu'
        The algorithm to use for structure learning. Options are:

        - ``'chow-liu'``: Learns a tree-structured DAG using maximum spanning
          tree over mutual information weights.
        - ``'tan'``: Learns a Tree-Augmented Naive Bayes structure. Requires
          ``class_node`` to be specified.

    root_node : str, int, or any hashable Python object, default=None
        The root node of the learned tree structure. Edges are directed away
        from this node in the final DAG. If ``None``, the node with the
        highest sum of edge weights is automatically selected as the root.

    class_node : str, int, or any hashable Python object, default=None
        The class variable for TAN. Directed edges are added from this node
        to all other nodes in the graph. Must be provided when
        ``estimator_type='tan'``. Ignored for ``'chow-liu'``.

    edge_weights_fn : str, default='mutual_info'
        The function used to compute edge weights between variable pairs.
        Options are:

        - ``'mutual_info'``: Standard mutual information.
        - ``'adjusted_mutual_info'``: Mutual information adjusted for chance.
        - ``'normalized_mutual_info'``: Mutual information normalized by
          the average of the entropies of the two variables.

    n_jobs : int, default=-1
        The number of parallel jobs for computing edge weights.
        ``-1`` means use all available processors.

    show_progress : bool, default=True
        If True, shows a progress bar while computing edge weights.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph after calling ``fit``.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features (variables) seen during ``fit``.

    feature_names_in_ : np.ndarray
        The feature names seen during ``fit``.

    Examples
    --------
    Learn a Chow-Liu tree structure from data:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.causal_discovery import TreeSearch
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(1000, 4)), columns=["A", "B", "C", "D"]
    ... )
    >>> ts = TreeSearch(estimator_type="chow-liu", root_node="A", show_progress=False)
    >>> ts.fit(data)
    >>> ts.causal_graph_.edges()

    Learn a TAN structure from data:

    >>> ts_tan = TreeSearch(
    ...     estimator_type="tan",
    ...     root_node="B",
    ...     class_node="A",
    ...     show_progress=False,
    ... )
    >>> ts_tan.fit(data)
    >>> ts_tan.causal_graph_.edges()

    References
    ----------
    .. [1] Chow, C. K.; Liu, C. N. (1968), "Approximating discrete probability
           distributions with dependence trees", IEEE Transactions on Information
           Theory, IT-14 (3): 462-467.
    .. [2] Friedman N, Geiger D and Goldszmidt M (1997). Bayesian network
           classifiers. Machine Learning 29: 131-163.
    """

    def __init__(
        self,
        estimator_type: str = "chow-liu",
        root_node: Optional[Hashable] = None,
        class_node: Optional[Hashable] = None,
        edge_weights_fn: str = "mutual_info",
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
        self.estimator_type = estimator_type
        self.root_node = root_node
        self.class_node = class_node
        self.edge_weights_fn = edge_weights_fn
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def _fit(self, X: pd.DataFrame):
        """
        Fit the TreeSearch algorithm to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The dataset to learn the causal structure from. Each column
            represents one variable.

        Returns
        -------
        self : pgmpy.causal_discovery.TreeSearch
            Returns the instance with fitted attributes set.
        """
        # Step 1: Validate arguments
        if self.estimator_type not in ("chow-liu", "tan"):
            raise ValueError(
                f"estimator_type must be one of: 'chow-liu', 'tan'. Got: {self.estimator_type}"
            )

        if self.estimator_type == "tan" and self.class_node is None:
            raise ValueError("class_node must be provided when estimator_type='tan'.")

        if self.root_node is not None and self.root_node not in X.columns:
            raise ValueError(f"root_node: {self.root_node} not found in data columns.")

        if self.class_node is not None and self.class_node not in X.columns:
            raise ValueError(
                f"class_node: {self.class_node} not found in data columns."
            )

        # Step 2: Delegate to the existing estimator
        est = TreeSearchEstimator(X, root_node=self.root_node, n_jobs=self.n_jobs)
        dag = est.estimate(
            estimator_type=self.estimator_type,
            class_node=self.class_node,
            edge_weights_fn=self.edge_weights_fn,
            show_progress=self.show_progress,
        )

        # Step 3: Store fitted attributes
        self.causal_graph_ = dag
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, weight=1, dtype="int"
        )

        return self
