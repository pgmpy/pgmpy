from typing import cast

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery


class BootstrapEstimator(BaseCausalDiscovery):
    """
    Bootstrap meta-estimator for causal discovery.

    This class wraps any causal discovery estimator to assess the stability and reliability of the learned causal
    structure. It repeatedly samples the input dataset with replacement, runs the base estimator on each bootstrap
    sample, and aggregates the resulting graphs to compute edge strengths and construct a robust consensus graph.
    This class implements the Non-Parametic Bootstrap technique which only work with provided sample data.

    Parameters
    ----------
    estimator : BaseCausalDiscovery instance
        The base causal discovery estimator to be wrapped (e.g., PC, HillClimbSearch, GES).

    n_bootstraps : int, default=10
        The number of bootstrap samples to generate and fit.

    sample_size : float, default=1.0
        The number of samples to draw from the input data for each bootstrap sample. Take a float from 0 to 1 as
        percentage.

    threshold : float, default=0.5
        The threshold for edge presence probability. Only edges that appear in at least this fraction of the bootstrap
        graphs are included in the final consensus graph. Must be between 0 and 1.

    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 means using all processors.

    show_progress : bool, default=True
        If True, shows a progress bar while fitting the bootstrap estimators.

    seed : int, default=None
        Seed for the random number generator to ensure reproducibility.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned robust consensus causal graph. The return type will be of
        type provided by the base estimator.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned consensus causal graph.

    edge_prob_ : pd.DataFrame
        DataFrame containing the estimated probabilities of edges across all bootstrap samples. Value of the cell
        indicates the frequency of the edge in the corresponding bootstrap sample.

    direction_prob_ : dict
        Dictionary mapping edge tuples (u, v) to the conditional probability of direction u -> v,
        given that an edge exists between u and v.

        Calculated as n1 / (n1 + n2 + n3), where:
        - n1: Number of bootstraps where u -> v is directed (only u -> v exists).
        - n2: Number of bootstraps where v -> u is directed (only v -> u exists).
        - n3: Number of bootstraps where u -- v is undirected (both u -> v and v -> u exist).

        Both keys (u, v) and (v, u) are present for any connected pair. The probability of the
        undirected edge u - v is 1 - direction_prob_[(u, v)] - direction_prob_[(v, u)].

    bootstrap_samples_ : list
        List of bootstrap samples. Each index contains a list of sample indexes.

    bootstrap_graphs_ : list
        List of graphs learned from the bootstrap samples.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        Names of the features in the input data.
    """

    def __init__(
        self,
        estimator: BaseCausalDiscovery,
        n_bootstraps: int = 10,
        sample_size: float = 1,
        threshold: float = 0.5,
        show_progress: bool = True,
        seed: int | None = None,
        n_jobs: int = -1,
    ):
        self.estimator = estimator
        self.n_bootstraps = n_bootstraps
        self.sample_size = sample_size
        self.threshold = threshold
        self.show_progress = show_progress
        self.seed = seed
        self.n_jobs = n_jobs

    @staticmethod
    def _bootstrap_iteration(
        X: pd.DataFrame,
        base_estimator: BaseCausalDiscovery,
        bootstrap_sample_size: int,
        seed: int | np.random.SeedSequence,
    ) -> tuple[list[int], BaseCausalDiscovery]:
        """Helper function to run a single bootstrap iteration."""

        # Create new sample by resampling
        rng = np.random.default_rng(seed)
        row_indices = list(rng.choice(X.index, size=bootstrap_sample_size, replace=True))
        sample = X.iloc[row_indices]

        # Fit the resample data on base estimator.
        base_estimator = cast(BaseCausalDiscovery, clone(base_estimator))
        est = base_estimator.fit(sample)
        return row_indices, est

    def _fit(self, X: pd.DataFrame):
        """
        Fit the bootstrap meta-estimator on the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to learn the causal structure from.
        """

        # Step 0: Initialize variables
        if not isinstance(self.estimator, BaseCausalDiscovery):
            raise ValueError("estimator must be an instance of BaseCausalDiscovery Class.")

        N = self.n_features_in_
        variables = self.feature_names_in_

        self.bootstrap_samples_ = list()
        self.bootstrap_graphs_ = list()

        seed_seq = np.random.SeedSequence(self.seed)
        child_seeds = seed_seq.spawn(self.n_bootstraps)

        bootstrap_sample_size = int(len(X) * self.sample_size)

        edge_presence = pd.DataFrame(
            np.zeros((N, N)),
            index=variables,
            columns=variables,
        )

        undirected_counts = pd.DataFrame(
            np.zeros((N, N)),
            index=variables,
            columns=variables,
        )

        # Step 1: Run bootstrap iterations
        results = cast(
            list[tuple[list[int], BaseCausalDiscovery]],
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._bootstrap_iteration)(X, self.estimator, bootstrap_sample_size, child_seeds[i])
                for i in trange(
                    self.n_bootstraps,
                    desc="Bootstrapping",
                    disable=not (self.show_progress and config.SHOW_PROGRESS),
                )
            ),
        )

        # Step 2: Aggregating the bootstrap results.
        for row_indices, est in results:
            causal_graph = est.causal_graph_
            adjacency_matrix = est.adjacency_matrix_

            self.bootstrap_samples_.append(row_indices)
            self.bootstrap_graphs_.append(causal_graph)

            edge_presence += adjacency_matrix
            undirected = (adjacency_matrix == 1) & (adjacency_matrix.T == 1)
            undirected_counts += undirected.astype(int)

        # Step 2.1: Calculate the direction probabilities
        rows, cols = np.where(edge_presence > 0)
        edges = list(zip(variables[rows], variables[cols]))

        self.direction_prob_ = {}
        for edge in edges:
            u, v = edge

            f_uv = undirected_counts.loc[u, v]
            f_utov = edge_presence.loc[u, v] - f_uv
            f_vtou = edge_presence.loc[v, u] - f_uv

            presence = f_uv + f_utov + f_vtou
            self.direction_prob_[edge] = f_utov / presence

        # Step 2.2: Calculate the edge probabilities
        self.edge_prob_ = edge_presence / self.n_bootstraps

        # Step 3: Form a consensus graph.
        self.causal_graph_ = self._estimate_consensus_graph(self.threshold)

        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, weight=1, dtype="int")

        return self

    def _estimate_consensus_graph(self, threshold: float) -> DAG | PDAG:

        variables = self.feature_names_in_

        if hasattr(self.estimator, "return_type"):
            return_type = self.estimator.return_type.lower()
        else:
            return_type = "dag"

        # Determine candidate edges based on edge probability and threshold
        rows, cols = np.where(self.edge_prob_ >= threshold)
        candidate_edges = [
            (
                variables[r],
                variables[c],
                self.edge_prob_.values[r, c],
            )
            for r, c in zip(rows, cols)
            if r != c
        ]

        # sort by descending probability, then alphabetical tie-breaker
        candidate_edges.sort(key=lambda x: (-x[2], x[0], x[1]))
        candidate_set = {(u, v) for u, v, _ in candidate_edges}

        if return_type in ("pdag", "cpdag"):
            # initialize consensus pdag
            pdag = PDAG()
            pdag.add_nodes_from(variables)

            processed_pairs = set()

            for u, v, _ in candidate_edges:
                pair = tuple(sorted((u, v)))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                # check if reverse orientation is present in candidate_edges
                if (v, u) in candidate_set:
                    p_utov = self.direction_prob_.get((u, v), 0.0)
                    p_vtou = self.direction_prob_.get((v, u), 0.0)
                    p_undirected = 1.0 - p_utov - p_vtou

                    if p_undirected >= p_utov and p_undirected >= p_vtou:
                        target = "undirected"
                    else:
                        target = "directed"
                else:
                    target = "directed"

                # add the edge and check for cyclicity
                if target == "undirected":
                    pdag.add_edge(u, v)
                    pdag.add_edge(v, u)
                    pdag.calibrate_directed_undirected_edges()
                    if not pdag.has_acyclic_extension():
                        pdag.remove_edge(u, v)
                        pdag.remove_edge(v, u)
                        pdag.calibrate_directed_undirected_edges()
                else:
                    # determine which direction to try first by comparing edge probabilities
                    prob_utov = self.edge_prob_.loc[u, v]
                    prob_vtou = self.edge_prob_.loc[v, u]

                    if prob_utov >= prob_vtou:
                        first = (u, v)
                    else:
                        first = (v, u)

                    x, y = first
                    pdag.add_edge(x, y)
                    pdag.calibrate_directed_undirected_edges()
                    if not pdag.has_acyclic_extension():
                        pdag.remove_edge(x, y)
                        pdag.calibrate_directed_undirected_edges()

                        # try opposite direction as backup only if it has support
                        if self.edge_prob_.loc[y, x] > 0:
                            pdag.add_edge(y, x)
                            pdag.calibrate_directed_undirected_edges()
                            if not pdag.has_acyclic_extension():
                                pdag.remove_edge(y, x)
                                pdag.calibrate_directed_undirected_edges()

            return pdag

        else:
            dag = DAG()
            dag.add_nodes_from(variables)

            for u, v, _ in candidate_edges:
                if not nx.has_path(dag, v, u):
                    dag.add_edge(u, v)

            return dag

    def get_causal_graph(self, threshold: float) -> DAG | PDAG:
        """
        Returns the consensus causal graph estimated using a specified edge probability threshold.

        Parameters
        ----------
        threshold : float
            The threshold for edge presence probability. Only edges that appear in at least
            this fraction of the bootstrap graphs are included. Must be between 0.0 and 1.0.

        Returns
        -------
        consensus_graph : DAG or PDAG
            The consensus causal graph (either a DAG or a PDAG/CPDAG depending on the return
            type of the base estimator).

        Examples
        --------
        >>> from pgmpy.causal_discovery import BootstrapEstimator, HillClimbSearch
        >>> from pgmpy.utils import get_example_model
        >>> data = get_example_model("asia").simulate(n_samples=100)
        >>> est = BootstrapEstimator(HillClimbSearch(data))
        >>> est.fit(data)
        >>> consensus_graph = est.get_causal_graph(threshold=0.3)
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0. Got {threshold} instead.")

        return self._estimate_consensus_graph(threshold)

    def get_adjacency_matrix(self, threshold: float) -> pd.DataFrame:
        """
        Returns the adjacency matrix of the consensus causal graph estimated using a specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for edge presence probability. Only edges that appear in at least
            this fraction of the bootstrap graphs are included. Must be between 0.0 and 1.0.

        Returns
        -------
        adjacency_matrix : pandas.DataFrame
            The adjacency matrix representation of the consensus causal graph.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0. Got {threshold} instead.")

        graph = self._estimate_consensus_graph(threshold)
        return nx.to_pandas_adjacency(graph, weight=1, dtype="int")
