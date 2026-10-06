from __future__ import annotations

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

    This class wraps any causal discovery estimator to assess the stability of the learned causal structure. It
    repeatedly samples the input dataset with replacement, runs the causal discovery estimator on each sampled dataset,
    records the edge between every pair of variables in each resulting graph, and constructs a consensus graph.

    The non-parametric bootstrap procedure operates as follows:
    1. Resamples the input dataset with replacement `n_bootstraps` times.
    2. Fits an independent clone of the base estimator on each resampled data.
    3. Records the type of edge between every pair of variables in each bootstrap graph (`bootstrap_edges_`).
    4. Constructs a consensus DAG or PDAG from the pairs of variables that have an edge in at least a `threshold`
    fraction of the bootstrap graphs.

    Parameters
    ----------
    estimator : BaseCausalDiscovery instance
        The base causal discovery estimator to be wrapped (e.g., PC, GES, HillClimbSearch).

    n_bootstraps : int, default=10
        The number of bootstrap samples to generate and fit. Must be at least 1.

    sample_size : float, default=1.0
        The fraction of samples to draw from the input data for each bootstrap sample. Must be greater than 0.0 and at
        most 1.0.

    threshold : float, default=0.5
        Only used to compute the consensus graph (`causal_graph_` and `adjacency_matrix_`); `bootstrap_edges_` doesn't
        depend on it. A pair of variables is connected in the consensus graph if it has an edge in at least this
        fraction of the bootstrap graphs, and in at least one of them. The edge gets its most frequent type in the
        bootstrap graphs; ties are left undirected in a PDAG and oriented from u to v in a DAG. Pairs are added from the
        most to the least frequent, and an edge that would create a cycle is skipped. Must be between 0.0 and 1.0.

    warm_start : bool, default=False
        When set to True, reuses existing fitted bootstrap graphs and adds new bootstrap samples upon subsequent calls
        to fit.

    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 uses all available processors.

    show_progress : bool, default=True
        If True, displays a progress bar during fitting.

    seed : int or None, default=None
        Seed for the random number generator to ensure reproducibility.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned consensus causal graph matching the return type of the base estimator.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the consensus causal graph.

    bootstrap_edges_ : pd.DataFrame
        The edge between every pair of variables in every bootstrap graph. Each row is a pair (u, v), with u before v
        in the order of the input features, and each column is a bootstrap graph, in the order of `bootstrap_samples_`.
        The values are "->" for u -> v, "<-" for v -> u, "--" for an undirected edge, and NaN if u and v aren't
        adjacent. Edge frequencies and other summaries can be computed with pandas, as shown in the examples.

    bootstrap_samples_ : np.ndarray
        2D array of shape `(n_bootstraps, bootstrap_sample_size)` containing the row indices used in each bootstrap
        sample.
        - Axis 0 (first axis): Represents each individual bootstrap sample.
        - Axis 1 (second axis): Represents the sampled row indices for that bootstrap sample.

    estimator_ : BaseCausalDiscovery
        The clone of ``estimator`` fitted on the first bootstrap sample.

    n_features_in_ : int
        The number of features in the input data.

    feature_names_in_ : np.ndarray
        Names of the features in the input data.

    Examples
    --------
    # Simulate dataset and fit BootstrapEstimator with HillClimbSearch:
    >>> from pgmpy.causal_discovery import BootstrapEstimator, HillClimbSearch
    >>> from pgmpy.example_models import load_model
    >>> data = load_model("bnlearn/cancer").simulate(n_samples=2000, seed=42)
    >>> data = data[sorted(data.columns)]
    >>> hc = HillClimbSearch(return_type="dag")
    >>> est = BootstrapEstimator(hc, seed=42, show_progress=False)
    >>> est = est.fit(data)

    # Show the edge between each pair of variables (rows) in each bootstrap graph (columns):
    >>> import pandas as pd
    >>> with pd.option_context("display.width", 1000):  # doctest: +NORMALIZE_WHITESPACE
    ...     print(est.bootstrap_edges_)
                           0    1    2    3    4    5    6    7    8    9
    u         v
    Cancer    Dyspnoea    ->   ->   ->   ->   ->  NaN   ->   ->   ->   ->
              Pollution   ->   ->   ->   ->   ->   ->  NaN   ->   ->   ->
              Smoker      ->   ->   ->   ->   <-   ->   ->  NaN   ->   ->
              Xray        <-   <-   <-   <-   ->   <-   <-   <-   <-   <-
    Dyspnoea  Pollution  NaN  NaN  NaN  NaN  NaN   <-  NaN  NaN  NaN  NaN
              Smoker     NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
              Xray       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
    Pollution Smoker     NaN  NaN  NaN  NaN  NaN  NaN  NaN   ->  NaN  NaN
              Xray       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
    Smoker    Xray       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN

    # Fraction of the bootstrap graphs in which each pair of variables is adjacent:
    >>> est.bootstrap_edges_.notna().mean(axis=1)  # doctest: +NORMALIZE_WHITESPACE
    u          v
    Cancer     Dyspnoea     0.9
               Pollution    0.9
               Smoker       0.9
               Xray         1.0
    Dyspnoea   Pollution    0.1
               Smoker       0.0
               Xray         0.0
    Pollution  Smoker       0.1
               Xray         0.0
    Smoker     Xray         0.0
    dtype: float64

    # Frequency of each edge type among the bootstrap graphs in which the pair is adjacent:
    >>> orientations = est.bootstrap_edges_.apply(lambda row: row.value_counts(normalize=True), axis=1)
    >>> orientations.round(2)  # doctest: +NORMALIZE_WHITESPACE
                           ->    <-   --
    u         v
    Cancer    Dyspnoea   1.00  0.00  0.0
              Pollution  1.00  0.00  0.0
              Smoker     0.89  0.11  0.0
              Xray       0.10  0.90  0.0
    Dyspnoea  Pollution  0.00  1.00  0.0
              Smoker      NaN   NaN  NaN
              Xray        NaN   NaN  NaN
    Pollution Smoker     1.00  0.00  0.0
              Xray        NaN   NaN  NaN
    Smoker    Xray        NaN   NaN  NaN

    # Show the adjacency matrix of the consensus graph:
    >>> est.adjacency_matrix_
               Cancer  Dyspnoea  Pollution  Smoker  Xray
    Cancer          0         1          1       1     0
    Dyspnoea        0         0          0       0     0
    Pollution       0         0          0       0     0
    Smoker          0         0          0       0     0
    Xray            1         0          0       0     0

    # Show the shapes of the bootstrap samples and bootstrap edges:
    >>> est.bootstrap_samples_.shape
    (10, 2000)
    >>> est.bootstrap_edges_.shape
    (10, 10)
    """

    _tags = {
        "name": "bootstrap_estimator",
        "data_types": ["discrete", "continuous", "mixed"],
        "identifiable_graph": "dag",
        "requires_target": False,
        "capability:multivariate": True,
        "capability:expert_knowledge": ["forbidden_edges", "required_edges", "search_space", "temporal_order"],
        "assumption:causal_sufficiency": False,
        "assumption:acyclicity": False,
        "assumption:faithfulness": False,
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
        "assumption:non_gaussian_noise": False,
        "assumption:low_noise": False,
    }

    def __init__(
        self,
        estimator: BaseCausalDiscovery,
        n_bootstraps: int = 10,
        sample_size: float = 1,
        threshold: float = 0.5,
        warm_start: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True,
        seed: int | None = None,
    ):

        self.estimator = estimator
        self.n_bootstraps = n_bootstraps
        self.sample_size = sample_size
        self.threshold = threshold
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.seed = seed

    @staticmethod
    def _bootstrap_iteration(
        X: pd.DataFrame,
        base_estimator: BaseCausalDiscovery,
        bootstrap_sample: np.ndarray,
    ) -> BaseCausalDiscovery:
        """
        Runs a single bootstrap iteration on resampled data.

        Selects rows from the dataset based on provided sample indices,
        creates a clone of the base estimator, and fits it on the resampled
        data.

        Parameters
        ----------
        X : pd.DataFrame
            The original input dataset.
        base_estimator : BaseCausalDiscovery
            The base causal discovery estimator to clone and fit.
        bootstrap_sample : np.ndarray
            Array of row indices defining the bootstrap resample.

        Returns
        -------
        fitted_estimator : BaseCausalDiscovery
            The fitted clone of the base estimator.
        """

        # Create new sample by resampling
        sample = X.iloc[bootstrap_sample]

        # Fit the resample data on base estimator.
        base_estimator = cast(BaseCausalDiscovery, clone(base_estimator))
        est = base_estimator.fit(sample)
        return est

    def _fit(self, X: pd.DataFrame) -> BootstrapEstimator:
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
        if self.n_bootstraps < 1:
            raise ValueError(f"n_bootstraps must be at least 1. Got {self.n_bootstraps} instead.")
        if not (0.0 < self.sample_size <= 1.0):
            raise ValueError(f"sample_size must be greater than 0.0 and at most 1.0. Got {self.sample_size} instead.")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be between 0.0 and 1.0. Got {self.threshold} instead.")

        if hasattr(self.estimator, "return_type"):
            self.return_type_ = self.estimator.return_type.lower()
        else:
            self.return_type_ = "dag"

        variables = self.feature_names_in_

        # Step 1: Resample dataset and fit base estimators
        rng = np.random.default_rng(self.seed)
        bootstrap_sample_size = int(len(X) * self.sample_size)

        # Generate all required bootstrap sample index arrays
        sample_indices = []
        for _ in range(self.n_bootstraps):
            sample_idx = rng.choice(len(X), size=bootstrap_sample_size, replace=True)
            sample_indices.append(sample_idx)
        all_samples = np.array(sample_indices)

        if self.warm_start and hasattr(self, "bootstrap_samples_"):
            if bootstrap_sample_size != self.bootstrap_samples_.shape[1]:
                raise ValueError("Cannot warm_start with a different dataset size.")
            if list(X.columns) != list(self.adjacency_matrix_.columns):
                raise ValueError("Cannot warm_start with a different dataset features.")

            n_existing = len(self.bootstrap_samples_)

            if self.n_bootstraps < n_existing:
                raise ValueError(
                    f"n_bootstraps={self.n_bootstraps} must be larger or equal to "
                    f"len(bootstrap_samples_)={n_existing} when warm_start==True."
                )
            samples_to_fit = all_samples[n_existing:]
        else:
            n_existing = 0
            samples_to_fit = all_samples

        # Only fit new bootstrap samples if needed. We skip this if warm_start is True and n_bootstraps has not
        # changed, but still form the consensus graph below in case class parameters (like threshold) were modified.
        if len(samples_to_fit) > 0:
            results = cast(
                list[BaseCausalDiscovery],
                Parallel(n_jobs=self.n_jobs)(
                    delayed(self._bootstrap_iteration)(X, self.estimator, samples_to_fit[i])
                    for i in trange(
                        len(samples_to_fit),
                        desc="Bootstrapping",
                        disable=not (self.show_progress and config.SHOW_PROGRESS),
                    )
                ),
            )

            # Record the edge type of every pair of variables (u, v), with u before v, in each new bootstrap graph:
            # "->" for u -> v, "<-" for v -> u, "--" for an undirected edge, and NaN if they aren't adjacent. An
            # undirected edge sets both entries of an adjacency matrix.
            graphs = np.array(
                [
                    est.adjacency_matrix_.reindex(index=variables, columns=variables, fill_value=0).values
                    for est in results
                ]
            )
            rows, cols = np.triu_indices(len(variables), k=1)
            u_to_v, v_to_u = graphs[:, rows, cols] == 1, graphs[:, cols, rows] == 1
            edge_types = np.select([u_to_v & v_to_u, u_to_v, v_to_u], ["--", "->", "<-"], default=None)
            new_edges = pd.DataFrame(
                edge_types.T,
                index=pd.MultiIndex.from_arrays([variables[rows], variables[cols]], names=["u", "v"]),
                columns=range(n_existing, n_existing + len(results)),
            ).astype(pd.CategoricalDtype(["->", "<-", "--"]))

            if n_existing > 0:
                self.bootstrap_samples_ = np.concatenate([self.bootstrap_samples_, samples_to_fit], axis=0)
                self.bootstrap_edges_ = pd.concat([self.bootstrap_edges_, new_edges], axis=1)
            else:
                self.bootstrap_samples_ = samples_to_fit
                self.bootstrap_edges_ = new_edges
                self.estimator_ = results[0]

        # Step 2: Form a consensus graph.
        self.causal_graph_ = self._estimate_consensus_graph(self.threshold)

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(encoding="binary", nodelist=variables)

        return self

    def _estimate_consensus_graph(self, threshold: float) -> DAG | PDAG:
        """
        Helper method to estimate the consensus graph for a given probability threshold.

        A pair of variables gets an edge if it has one in at least a `threshold` fraction of the bootstrap graphs, and
        in at least one of them. The edge gets its most frequent type in the bootstrap graphs; ties are left undirected
        in a PDAG and oriented from u to v in a DAG. Pairs are added from the most to the least frequent, and an edge
        that would create a cycle is skipped.

        Parameters
        ----------
        threshold : float
            Edge presence probability threshold (between 0.0 and 1.0).

        Returns
        -------
        consensus_graph : DAG or PDAG
            The constructed consensus causal graph.
        """
        edges = self.bootstrap_edges_
        counts = pd.DataFrame({edge_type: (edges == edge_type).sum(axis=1) for edge_type in ("->", "<-", "--")})
        frequency = counts.sum(axis=1) / edges.shape[1]
        candidates = frequency[(frequency >= threshold) & (frequency > 0)].sort_values(ascending=False, kind="stable")

        is_pdag = self.return_type_ in ("pdag", "cpdag")
        graph = PDAG() if is_pdag else DAG()
        graph.add_nodes_from(self.feature_names_in_)

        for u, v in candidates.index:
            n_u_to_v, n_v_to_u, n_undirected = counts.loc[(u, v)]

            if is_pdag and (n_undirected >= max(n_u_to_v, n_v_to_u) or n_u_to_v == n_v_to_u):
                graph.add_edge(u, v, "--")
                continue

            x, y = (u, v) if n_u_to_v >= n_v_to_u else (v, u)
            if is_pdag:
                graph.add_edge(x, y, "->")
                if not graph.has_acyclic_extension():
                    graph.remove_edge(x, y, "->")
            elif not nx.has_path(graph, y, x):
                graph.add_edge(x, y)

        return graph

    def get_consensus_graph(self, threshold: float) -> DAG | PDAG:
        """
        Returns the consensus causal graph estimated using a specified edge
        probability threshold.

        Parameters
        ----------
        threshold : float
            The threshold for edge presence probability. Only pairs of variables that have an edge in at least this
            fraction of the bootstrap graphs, and in at least one of them, are connected. See the `threshold` parameter
            of the class for how the edges are oriented. Must be between 0.0 and 1.0.

        Returns
        -------
        consensus_graph : DAG or PDAG
            The consensus causal graph (either a DAG or a PDAG depending on
            the return type of the base estimator).

        Raises
        ------
        ValueError
            If `threshold` is not between 0.0 and 1.0.

        Examples
        --------
        # Fit BootstrapEstimator on dataset:
        >>> from pgmpy.causal_discovery import BootstrapEstimator, HillClimbSearch
        >>> from pgmpy.example_models import load_model
        >>> data = load_model("bnlearn/cancer").simulate(n_samples=2000, seed=42)
        >>> data = data[sorted(data.columns)]
        >>> hc = HillClimbSearch(return_type="dag")
        >>> est = BootstrapEstimator(hc, seed=42, show_progress=False)
        >>> est = est.fit(data)

        # Extract consensus graph with threshold 0.4:
        >>> consensus_graph = est.get_consensus_graph(threshold=0.4)
        >>> sorted(consensus_graph.edges())
        [('Cancer', 'Dyspnoea'), ('Cancer', 'Pollution'), ('Cancer', 'Smoker'), ('Xray', 'Cancer')]
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0. Got {threshold} instead.")

        return self._estimate_consensus_graph(threshold)

    def get_adjacency_matrix(self, threshold: float) -> pd.DataFrame:
        """
        Returns the adjacency matrix of the consensus causal graph estimated
        using a specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for edge presence probability. Only pairs of variables that have an edge in at least this
            fraction of the bootstrap graphs, and in at least one of them, are connected. See the `threshold` parameter
            of the class for how the edges are oriented. Must be between 0.0 and 1.0.

        Returns
        -------
        adjacency_matrix : pandas.DataFrame
            The adjacency matrix representation of the consensus causal graph.

        Raises
        ------
        ValueError
            If `threshold` is not between 0.0 and 1.0.

        Examples
        --------
        # Fit BootstrapEstimator on dataset:
        >>> from pgmpy.causal_discovery import BootstrapEstimator, HillClimbSearch
        >>> from pgmpy.example_models import load_model
        >>> data = load_model("bnlearn/cancer").simulate(n_samples=2000, seed=42)
        >>> data = data[sorted(data.columns)]
        >>> hc = HillClimbSearch(return_type="dag")
        >>> est = BootstrapEstimator(hc, seed=42, show_progress=False)
        >>> est = est.fit(data)

        # Extract adjacency matrix with threshold 0.4:
        >>> adj_matrix = est.get_adjacency_matrix(threshold=0.4)
        >>> adj_matrix
                   Cancer  Dyspnoea  Pollution  Smoker  Xray
        Cancer          0         1          1       1     0
        Dyspnoea        0         0          0       0     0
        Pollution       0         0          0       0     0
        Smoker          0         0          0       0     0
        Xray            1         0          0       0     0
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0. Got {threshold} instead.")

        graph = self._estimate_consensus_graph(threshold)
        return graph.to_adjacency(encoding="binary", nodelist=self.feature_names_in_)
