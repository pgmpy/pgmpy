"""Deprecated compatibility shims for :class:`pgmpy.causal_discovery.ChowLiu` / :class:`pgmpy.causal_discovery.TAN`."""

import warnings

from pgmpy.causal_discovery import TAN as _TAN
from pgmpy.causal_discovery import ChowLiu as _ChowLiu
from pgmpy.estimators import StructureEstimator


class TreeSearch(StructureEstimator):
    """
    Deprecated: use :class:`pgmpy.causal_discovery.ChowLiu` or
    :class:`pgmpy.causal_discovery.TAN` instead.

    Search class for learning tree-related graph structures. The algorithms
    supported are Chow-Liu and Tree-augmented naive bayes (TAN); both delegate
    to their canonical implementations in `pgmpy.causal_discovery`.

    Parameters
    ----------
    data: pandas.DataFrame object
        dataframe object where each column represents one variable.

    root_node: str, int, or any hashable python object, optional
        The root node of the tree structure. If None, then root node is
        auto-picked as the node with the highest sum of edge weights.

    n_jobs: int (default: -1)
        Number of jobs to run in parallel. `-1` means use all processors.
    """

    def __init__(self, data, root_node=None, n_jobs=-1, **kwargs):
        warnings.warn(
            "TreeSearch is deprecated and will be removed in v2.0. Please use "
            "pgmpy.causal_discovery.ChowLiu or pgmpy.causal_discovery.TAN instead.",
            FutureWarning,
            stacklevel=2,
        )
        if root_node is not None and root_node not in data.columns:
            raise ValueError(f"Root node: {root_node} not found in data columns.")

        self.root_node = root_node
        self.n_jobs = n_jobs
        super().__init__(data, **kwargs)

    # Legacy private helpers, forwarded to the canonical tree-search mixin.
    _get_weights = staticmethod(_ChowLiu._get_weights)
    _get_conditional_weights = staticmethod(_TAN._get_conditional_weights)
    _create_tree_and_dag = staticmethod(_ChowLiu._create_tree_and_dag)

    def estimate(
        self,
        estimator_type="chow-liu",
        class_node=None,
        edge_weights_fn="mutual_info",
        show_progress=True,
    ):
        """
        Estimate the `DAG` structure that fits best to the given data set
        without parametrization. Delegates to
        :class:`pgmpy.causal_discovery.ChowLiu` (`estimator_type="chow-liu"`)
        or :class:`pgmpy.causal_discovery.TAN` (`estimator_type="tan"`); refer
        to their documentation for parameter details.

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            The estimated model structure.
        """
        if estimator_type == "chow-liu":
            est = _ChowLiu(
                root_node=self.root_node,
                edge_weights_fn=edge_weights_fn,
                n_jobs=self.n_jobs,
                show_progress=show_progress,
            )
        elif estimator_type == "tan":
            est = _TAN(
                class_node=class_node,
                root_node=self.root_node,
                edge_weights_fn=edge_weights_fn,
                n_jobs=self.n_jobs,
                show_progress=show_progress,
            )
        else:
            raise ValueError(f"`estimator_type` must be one of: chow-liu, tan. Got: {estimator_type}")

        return est.fit(self.data).causal_graph_
