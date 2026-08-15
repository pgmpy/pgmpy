from collections.abc import Callable
from itertools import islice, permutations
from math import factorial

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.ci_tests import get_ci_test


class SP(BaseCausalDiscovery):
    """
    The Sparsest Permutation (SP) algorithm exhaustively searches over all permutations of the variables. For each
    permutation, it constructs the corresponding minimal independence map (I-MAP) using conditional independence tests
    and returns either that DAG or its PDAG representation, depending on `return_type`.

    The algorithm is statistically consistent under the Sparsest Markov Representation (SMR) assumption, which is weaker
    than the restricted faithfulness assumption required by many constraint-based methods.

    Parameters
    ----------
    ci_test : str or callable, default=None
        Conditional independence test to use for constructing the minimal I-MAP. This can be any of the CI test
        implemented in :mod:`pgmpy.ci_tests` or a custom function that follows the signature of the built-in CI tests.

        If None, the appropriate CI test will be chosen based on the data type.

    significance_level : float, default=0.01
        Significance level used by the conditional independence test.

    max_iter : int or None, default=None
        Maximum number of permutations to evaluate. If None, all possible permutations are considered.

    return_type : str, default='dag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    seed : int or None, default=None
        Seed for the random number generator used to shuffle the variables before searching over permutations.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph as a directed acyclic graph (DAG) or partially directed acyclic graph (PDAG).

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    optimal_permutations_ : list[tuple]
        All permutations that produce a DAG with the minimum number of edges.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/cancer")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the SP algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import SP
    >>> sp = SP(ci_test="chi_square")
    >>> sp.fit(df)
    SP(ci_test='chi_square')
    >>> sp.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> sp.n_features_in_
    5

    References
    ----------
    - :footcite:t:`raskutti2019learningdirectedacyclicgraphs`
    """

    def __init__(
        self,
        ci_test: str | Callable | None = None,
        significance_level: float = 0.01,
        max_iter: int | None = None,
        return_type: str = "dag",
        show_progress: bool = True,
        seed: int | None = None,
    ):
        self.ci_test = ci_test
        self.significance_level = significance_level
        self.max_iter = max_iter
        self.return_type = return_type
        self.show_progress = show_progress
        self.seed = seed

    def _build_imap_edges(
        self,
        permutation: tuple[str, ...],
        n_edge_limit: int | float = np.inf,
    ) -> list[tuple[str, str]] | None:
        """
        Construct the edges of the minimal I-MAP for a given variable ordering.

        For each variable in the permutation (skipping the first, which has no predecessors and therefore contributes no
        edges), every preceding variable is considered as a candidate parent. An edge is added from a predecessor to the
        current variable if the two variables are conditionally dependent given all other predecessors of the current
        variable.

        Parameters
        ----------
        permutation : tuple of str
            A permutation of the variable names.

        n_edge_limit : int, default=np.inf
            Maximum number of edges allowed during construction. Returns None if the edge count exceeds this value,
            enabling early pruning of unpromising permutations.

        Returns
        -------
        list[tuple[str, str]] or None
            The edges of the minimal I-MAP, or None if the search was aborted early because `n_edge_limit` was exceeded.
        """
        edges = []

        for node_idx in range(1, len(permutation)):
            node = permutation[node_idx]
            predecessors = permutation[:node_idx]
            for predecessor in predecessors:
                conditioning_nodes = (p for p in predecessors if p != predecessor)
                independent = self.ci_test_(
                    X=predecessor,
                    Y=node,
                    Z=conditioning_nodes,
                    significance_level=self.significance_level,
                )
                if not independent:
                    edges.append((predecessor, node))
                    if len(edges) > n_edge_limit:
                        return None

        return edges

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the SP algorithm.

        Parameters
        ----------
        X : pandas.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.SP
            Returns the instance with the fitted attributes.
        """
        # Step 0: Check inputs
        if self.max_iter is not None and self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1 to evaluate at least one permutation, got {self.max_iter}.")
        if self.return_type.lower() not in ("dag", "pdag"):
            raise ValueError(f"return_type must be one of: dag, pdag. Got: {self.return_type}")

        # Step 1: Initialize variables and data structures.
        self.ci_test_ = get_ci_test(test=self.ci_test, data=X)
        nodes = list(self.feature_names_in_)

        rng = np.random.default_rng(self.seed)
        rng.shuffle(nodes)

        # Step 2: Run the iterations.
        min_edges = np.inf
        best_ordering = None
        best_edges = None
        optimal_permutations = []

        max_permutations = factorial(len(nodes))
        n_iterations = min(self.max_iter, max_permutations) if self.max_iter is not None else max_permutations
        permutation_iter = islice(permutations(nodes), n_iterations)

        for permutation in tqdm(
            permutation_iter,
            total=n_iterations,
            desc="Searching over permutations",
            disable=not (self.show_progress and config.SHOW_PROGRESS),
        ):
            edges = self._build_imap_edges(
                permutation,
                n_edge_limit=min_edges,
            )
            if edges is None:
                continue

            n_edges = len(edges)

            # If new graph with minimum edges is found, it restarts the list of optimal permutations
            if n_edges < min_edges:
                min_edges = n_edges
                best_ordering = permutation
                best_edges = edges
                optimal_permutations = [permutation]
            # If the graph is tied with current minimum, it adds it to the list
            elif n_edges == min_edges:
                optimal_permutations.append(permutation)

        self.optimal_permutations_ = optimal_permutations

        # Step 3: Construct the DAG using the optimal permutation and assign attributes.
        current_model = DAG()
        current_model.add_nodes_from(best_ordering)
        current_model.add_edges_from(best_edges)

        if self.return_type.lower() == "dag":
            self.causal_graph_ = current_model
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = current_model.to_pdag()

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self
