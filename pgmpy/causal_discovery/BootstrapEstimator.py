from __future__ import annotations

from typing import cast

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.utils import get_tags
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import ADMG, DAG, MAG, PDAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.utils._warnings import _warn_external


class BootstrapEstimator(BaseCausalDiscovery):
    """
    Bootstrap meta-estimator for causal discovery.

    This class wraps any causal discovery estimator to assess the stability of the learned causal structure. It
    repeatedly samples the input dataset, runs the causal discovery estimator on each sampled dataset, records the edge
    between every pair of variables in each resulting graph, and constructs a consensus graph.

    The procedure operates as follows:
    1. Draws `n_bootstraps` samples of the input dataset, each with a `sample_fraction` of its rows, either with
    replacement (the non-parametric bootstrap) or, with ``replace=False``, without replacement (subsampling).
    2. Fits an independent clone of the base estimator on each sample, and one more on the full data (`estimator_`).
    3. Records the type of edge between every pair of variables in each bootstrap graph (`bootstrap_edges_`).
    4. Constructs a consensus graph, of the same class as the bootstrap graphs, from the pairs of variables that have an
    edge in at least a `threshold` fraction of the bootstrap graphs.

    Parameters
    ----------
    estimator : BaseCausalDiscovery instance
        The base causal discovery estimator to be wrapped (e.g., PC, GES, HillClimbSearch).

    n_bootstraps : int, default=100
        The number of bootstrap samples to generate and fit. Must be at least 1.

    sample_fraction : float, default=1.0
        The fraction of the rows of the input data to draw for each bootstrap sample. Must be greater than 0.0 and at
        most 1.0.

    replace : bool, default=True
        If True, each bootstrap sample is drawn with replacement, so rows can repeat. If False, each sample is a subset
        of distinct rows, which requires a `sample_fraction` below 1.0.

    threshold : float, default=0.5
        Edge frequency threshold for the consensus graph (`causal_graph_`, `adjacency_matrix_`); it doesn't affect
        `bootstrap_edges_`. Pairs with an edge in at least this fraction of the bootstrap graphs are connected with
        their most frequent edge type; ties become undirected in a PDAG and otherwise take the first tied category, e.g.
        u -> v in a DAG. Pairs are added from most to least frequent, skipping with a warning any edge that would create
        a directed cycle. Must be between 0.0 and 1.0.

    warm_start : bool, default=False
        When set to True, reuses existing fitted bootstrap graphs and adds new bootstrap samples upon subsequent calls
        to fit.

    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 uses all available processors.

    show_progress : bool, default=True
        If True, displays a progress bar during fitting. The wrapped estimator's own progress bars are always turned
        off.

    seed : int or None, default=None
        Seed for the random number generator to ensure reproducibility.

    Attributes
    ----------
    causal_graph_ : DAG, PDAG, MAG, or PAG
        The learned consensus causal graph, of the same class as the bootstrap graphs.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the consensus causal graph: binary for a DAG or PDAG, and the edge types
        of ``to_adjacency(encoding="edge_type")`` otherwise.

    bootstrap_edges_ : pd.DataFrame
        The edge between every pair of variables in every bootstrap graph. Each row is a pair (u, v), and each column is
        a bootstrap graph. The values are the edge types oriented from u to v, e.g. "->" for u -> v, "<-" for v -> u,
        "--" for an undirected edge, and NaN if u and v aren't adjacent. Edge frequencies and other summaries can be
        computed with pandas, as shown in the examples.

    bootstrap_samples_ : np.ndarray
        2D array of shape `(n_bootstraps, int(sample_fraction * n_rows))` containing the row indices used in each
        bootstrap sample.
        - Axis 0 (first axis): Represents each individual bootstrap sample.
        - Axis 1 (second axis): Represents the sampled row indices for that bootstrap sample.

    estimator_ : BaseCausalDiscovery
        A clone of ``estimator`` fitted on the full input data, e.g. to compare its graph with the bootstrap edges.

    n_features_in_ : int
        The number of features in the input data.

    feature_names_in_ : np.ndarray
        Names of the features in the input data.

    Examples
    --------
    # Simulate a dataset and fit BootstrapEstimator with PC, which returns a CPDAG for each bootstrap sample:
    >>> from pgmpy.causal_discovery import PC, BootstrapEstimator
    >>> from pgmpy.example_models import load_model
    >>> data = load_model("bnlearn/cancer").simulate(n_samples=1000, seed=42)
    >>> data = data[sorted(data.columns)]
    >>> pc = PC(return_type="pdag")
    >>> est = BootstrapEstimator(pc, n_bootstraps=10, seed=42, show_progress=False)
    >>> est = est.fit(data)

    # Show the edge between each pair of variables (rows) in each bootstrap graph (columns):
    >>> import pandas as pd
    >>> with pd.option_context("display.width", 1000):  # doctest: +NORMALIZE_WHITESPACE
    ...     print(est.bootstrap_edges_)
                           0    1    2    3    4    5    6    7    8    9
    u         v
    Cancer    Dyspnoea    <-  NaN  NaN   <-  NaN  NaN  NaN  NaN  NaN  NaN
              Pollution  NaN  NaN   <-  NaN  NaN  NaN  NaN   <-  NaN  NaN
              Smoker      <-  NaN  NaN   <-   <-   --   <-   <-   <-   <-
              Xray        <-   --   <-   <-   <-   --   <-   <-   <-   <-
    Dyspnoea  Pollution  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
              Smoker     NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
              Xray       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
    Pollution Smoker     NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
              Xray       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
    Smoker    Xray       NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN

    # Fraction of the bootstrap graphs in which each pair of variables is adjacent:
    >>> est.bootstrap_edges_.notna().mean(axis=1)  # doctest: +NORMALIZE_WHITESPACE
    u          v
    Cancer     Dyspnoea     0.2
               Pollution    0.2
               Smoker       0.8
               Xray         1.0
    Dyspnoea   Pollution    0.0
               Smoker       0.0
               Xray         0.0
    Pollution  Smoker       0.0
               Xray         0.0
    Smoker     Xray         0.0
    dtype: float64

    # Frequency of each edge type among the bootstrap graphs in which the pair is adjacent:
    >>> orientations = est.bootstrap_edges_.apply(lambda row: row.value_counts(normalize=True), axis=1)
    >>> orientations.round(2)  # doctest: +NORMALIZE_WHITESPACE
                           --    <-
    u         v
    Cancer    Dyspnoea   0.00  1.00
              Pollution  0.00  1.00
              Smoker     0.12  0.88
              Xray       0.20  0.80
    Dyspnoea  Pollution   NaN   NaN
              Smoker      NaN   NaN
              Xray        NaN   NaN
    Pollution Smoker      NaN   NaN
              Xray        NaN   NaN
    Smoker    Xray        NaN   NaN

    # Show the adjacency matrix of the consensus graph:
    >>> est.adjacency_matrix_
               Cancer  Dyspnoea  Pollution  Smoker  Xray
    Cancer          0         0          0       0     0
    Dyspnoea        0         0          0       0     0
    Pollution       0         0          0       0     0
    Smoker          1         0          0       0     0
    Xray            1         0          0       0     0

    # Show the shapes of the bootstrap samples and bootstrap edges:
    >>> est.bootstrap_samples_.shape
    (10, 1000)
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
        n_bootstraps: int = 100,
        sample_fraction: float = 1,
        replace: bool = True,
        threshold: float = 0.5,
        warm_start: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True,
        seed: int | None = None,
    ):

        self.estimator = estimator
        self.n_bootstraps = n_bootstraps
        self.sample_fraction = sample_fraction
        self.replace = replace
        self.threshold = threshold
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.seed = seed

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if isinstance(self.estimator, BaseCausalDiscovery):
            tags.input_tags = get_tags(self.estimator).input_tags
        return tags

    @staticmethod
    def _bootstrap_iteration(
        X: pd.DataFrame,
        base_estimator: BaseCausalDiscovery,
        bootstrap_sample: np.ndarray,
    ) -> DAG | PDAG | MAG | ADMG:
        """
        Fits a clone of the base estimator on a bootstrap sample of the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            The original input dataset.
        base_estimator : BaseCausalDiscovery
            The base causal discovery estimator to clone and fit.
        bootstrap_sample : np.ndarray
            Array of row indices to fit on.

        Returns
        -------
        causal_graph : DAG, PDAG, MAG, or ADMG
            The causal graph learned by the fitted clone.
        """
        sample = X.iloc[bootstrap_sample]
        base_estimator = cast(BaseCausalDiscovery, clone(base_estimator))
        est = base_estimator.fit(sample)
        return est.causal_graph_

    def _fit(self, X: pd.DataFrame) -> BootstrapEstimator:
        """
        Fit the bootstrap meta-estimator on the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to learn the causal structure from.
        """

        # Step 1: Initial checks for arguments
        # Step 1.1: Check the parameters
        if not isinstance(self.estimator, BaseCausalDiscovery):
            raise ValueError("estimator must be an instance of BaseCausalDiscovery Class.")
        if self.n_bootstraps < 1:
            raise ValueError(f"n_bootstraps must be at least 1. Got {self.n_bootstraps} instead.")
        if not (0.0 < self.sample_fraction <= 1.0):
            raise ValueError(
                f"sample_fraction must be greater than 0.0 and at most 1.0. Got {self.sample_fraction} instead."
            )
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be between 0.0 and 1.0. Got {self.threshold} instead.")
        if not self.replace and self.sample_fraction == 1.0:
            raise ValueError(
                "replace=False with sample_fraction=1.0 uses the full data in every bootstrap sample. Use a "
                "sample_fraction below 1.0."
            )

        # Step 1.2: Check that a warm start can continue the earlier fit
        bootstrap_sample_size = int(len(X) * self.sample_fraction)
        warm_start = self.warm_start and hasattr(self, "bootstrap_samples_")
        if warm_start:
            if bootstrap_sample_size != self.bootstrap_samples_.shape[1]:
                raise ValueError(
                    "Cannot warm_start with a different dataset size or sample_fraction: the existing bootstrap "
                    f"samples have {self.bootstrap_samples_.shape[1]} rows, the new ones would have "
                    f"{bootstrap_sample_size}."
                )
            if list(X.columns) != list(self.adjacency_matrix_.columns):
                raise ValueError("Cannot warm_start with a different dataset features.")
            if self.n_bootstraps < len(self.bootstrap_samples_):
                raise ValueError(
                    f"n_bootstraps={self.n_bootstraps} must be larger or equal to "
                    f"len(bootstrap_samples_)={len(self.bootstrap_samples_)} when warm_start==True."
                )

        # Step 2: Fit the estimator on the bootstrap samples
        # Step 2.1: Draw the bootstrap samples
        rng = np.random.default_rng(self.seed)
        all_samples = np.array(
            [rng.choice(len(X), size=bootstrap_sample_size, replace=self.replace) for _ in range(self.n_bootstraps)]
        )

        # Step 2.2: Without a warm start, initialize the results and fit `estimator_` on the full data
        rows, cols = np.triu_indices(len(self.feature_names_in_), k=1)
        if not warm_start:
            self.bootstrap_samples_ = np.empty((0, bootstrap_sample_size), dtype=all_samples.dtype)
            self.bootstrap_edges_ = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [self.feature_names_in_[rows], self.feature_names_in_[cols]], names=["u", "v"]
                )
            )
            self.estimator_ = clone(self.estimator).fit(X)

        n_existing = len(self.bootstrap_samples_)
        samples_to_fit = all_samples[n_existing:]

        # Step 2.3: Fit the estimator on the new bootstrap samples
        if len(samples_to_fit) > 0:
            graphs = Parallel(n_jobs=self.n_jobs)(
                delayed(self._bootstrap_iteration)(X, self.estimator, indices)
                for indices in tqdm(
                    samples_to_fit,
                    desc="Bootstrapping",
                    disable=not (self.show_progress and config.SHOW_PROGRESS),
                )
            )

            # Step 2.4: Record the edge types of the new bootstrap graphs
            edge_types = []
            for graph in graphs:
                cells = graph.to_adjacency(encoding="edge_type", nodelist=self.feature_names_in_).to_numpy()[rows, cols]
                edge_types.append([",".join(c) if isinstance(c, tuple) else (None if c == 0 else c) for c in cells])
            new_edges = pd.DataFrame(
                np.array(edge_types, dtype=object).T,
                index=self.bootstrap_edges_.index,
                columns=range(n_existing, n_existing + len(graphs)),
            )

            self.bootstrap_samples_ = np.concatenate([self.bootstrap_samples_, samples_to_fit], axis=0)
            edges = pd.concat([self.bootstrap_edges_.astype(object), new_edges], axis=1)
            self.bootstrap_edges_ = edges.astype(pd.CategoricalDtype(sorted(set(edges.stack().dropna()))))

        # Step 3: Store results
        # Step 3.1: Form the consensus graph
        self.causal_graph_ = self.get_consensus_graph(self.threshold)

        # Step 3.2: Compute its adjacency matrix, binary for DAGs and PDAGs
        encoding = "binary" if isinstance(self.causal_graph_, (DAG, PDAG)) else "edge_type"
        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(encoding=encoding, nodelist=self.feature_names_in_)

        return self

    def get_consensus_graph(self, threshold: float) -> DAG | PDAG | MAG | ADMG:
        """
        Returns the consensus causal graph estimated using a specified edge
        frequency threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the edge frequency. Only pairs of variables that have an edge in at least this
            fraction of the bootstrap graphs, and in at least one of them, are connected. See the `threshold` parameter
            of the class for how the edges are oriented. Must be between 0.0 and 1.0.

        Returns
        -------
        consensus_graph : DAG, PDAG, MAG, or ADMG
            The consensus causal graph, of the same class as the bootstrap graphs.

        Raises
        ------
        ValueError
            If `threshold` is not between 0.0 and 1.0.

        Examples
        --------
        # Fit BootstrapEstimator on dataset:
        >>> from pgmpy.causal_discovery import PC, BootstrapEstimator
        >>> from pgmpy.example_models import load_model
        >>> data = load_model("bnlearn/cancer").simulate(n_samples=1000, seed=42)
        >>> data = data[sorted(data.columns)]
        >>> pc = PC(return_type="pdag")
        >>> est = BootstrapEstimator(pc, n_bootstraps=10, seed=42, show_progress=False)
        >>> est = est.fit(data)

        # Extract consensus graph with threshold 0.2:
        >>> consensus_graph = est.get_consensus_graph(threshold=0.2)
        >>> sorted(consensus_graph.directed_edges)
        [('Dyspnoea', 'Cancer'), ('Pollution', 'Cancer'), ('Smoker', 'Cancer'), ('Xray', 'Cancer')]
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0. Got {threshold} instead.")

        edges = self.bootstrap_edges_
        counts = pd.DataFrame(
            {edge_type: (edges == edge_type).sum(axis=1) for edge_type in edges.iloc[:, 0].cat.categories}
        )
        frequency = edges.notna().mean(axis=1)
        candidates = frequency[(frequency >= threshold) & (frequency > 0)].sort_values(ascending=False, kind="stable")

        graph = type(self.estimator_.causal_graph_)()
        graph.add_nodes_from(self.feature_names_in_)

        skipped = []
        for u, v in candidates.index:
            pair_counts = counts.loc[(u, v)]
            most_frequent = pair_counts.index[pair_counts == pair_counts.max()]
            edge_types = ["--"] if isinstance(graph, PDAG) and len(most_frequent) > 1 else most_frequent[0].split(",")

            # A directed edge x -> y creates a directed cycle if there already is a directed path from y to x.
            directed = [
                (u, v) if edge_type == "->" else (v, u) for edge_type in edge_types if edge_type in ("->", "<-")
            ]
            if isinstance(graph, DAG):
                if any(nx.has_path(graph, y, x) for x, y in directed):
                    skipped.append((u, v))
                else:
                    graph.add_edges_from(directed)
            elif any(graph.has_path(y, x, edge_types="->") for x, y in directed):
                skipped.append((u, v))
            else:
                graph.add_edges_from([(u, v, edge_type) for edge_type in edge_types])

        if skipped:
            _warn_external(
                f"{len(skipped)} pair(s) of variables passed the threshold but were left out of the consensus graph "
                f"because their edge would create a directed cycle: {skipped}"
            )

        return graph

    def get_adjacency_matrix(self, threshold: float) -> pd.DataFrame:
        """
        Returns the adjacency matrix of the consensus causal graph estimated
        using a specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the edge frequency. Only pairs of variables that have an edge in at least this
            fraction of the bootstrap graphs, and in at least one of them, are connected. See the `threshold` parameter
            of the class for how the edges are oriented. Must be between 0.0 and 1.0.

        Returns
        -------
        adjacency_matrix : pandas.DataFrame
            The adjacency matrix representation of the consensus causal graph: binary for a DAG or PDAG, and the edge
            types of ``to_adjacency(encoding="edge_type")`` otherwise.

        Raises
        ------
        ValueError
            If `threshold` is not between 0.0 and 1.0.

        Examples
        --------
        # Fit BootstrapEstimator on dataset:
        >>> from pgmpy.causal_discovery import PC, BootstrapEstimator
        >>> from pgmpy.example_models import load_model
        >>> data = load_model("bnlearn/cancer").simulate(n_samples=1000, seed=42)
        >>> data = data[sorted(data.columns)]
        >>> pc = PC(return_type="pdag")
        >>> est = BootstrapEstimator(pc, n_bootstraps=10, seed=42, show_progress=False)
        >>> est = est.fit(data)

        # Extract adjacency matrix with threshold 0.2:
        >>> adj_matrix = est.get_adjacency_matrix(threshold=0.2)
        >>> adj_matrix
                   Cancer  Dyspnoea  Pollution  Smoker  Xray
        Cancer          0         0          0       0     0
        Dyspnoea        1         0          0       0     0
        Pollution       1         0          0       0     0
        Smoker          1         0          0       0     0
        Xray            1         0          0       0     0
        """
        graph = self.get_consensus_graph(threshold)
        encoding = "binary" if isinstance(graph, (DAG, PDAG)) else "edge_type"
        return graph.to_adjacency(encoding=encoding, nodelist=self.feature_names_in_)
