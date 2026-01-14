from itertools import chain, combinations
from typing import (
    Callable,
    Collection,
    Dict,
    Hashable,
    Optional,
    Set,
    Tuple,
    Union,
)

import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import UndirectedGraph
from pgmpy.estimators import ExpertKnowledge
from pgmpy.estimators.CITests import ci_registry
from pgmpy.global_vars import logger
from pgmpy.independencies import Independencies


class _BaseCausalDiscovery(BaseEstimator):
    """
    Base class for all causal discovery estimators in pgmpy.

    Sets the sklearn tags and defines a method to check the input data for fitting.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = False
        tags.input_tags.positive_only = False
        tags.target_tags.required = False
        return tags

    def _check_fit_data(self, X):
        """Check the input data for fitting the causal discovery algorithm.

        Parameters
        ----------
        X: pd.DataFrame
            The data to fit the causal discovery algorithm on.
        """
        n_samples, n_features = X.shape

        if n_features == 0:
            raise ValueError(
                f"0 feature(s) (shape={X.shape}) while a minimum of 1 is required."
            )
        if n_samples < 2:
            raise ValueError(f"n_samples = {n_samples}, at least 2 are required.")

        # Handle cases like complex data, sparse arrays etc. first
        validate_data(
            self,
            X=X,
            dtype=None,
            accept_sparse=False,
            ensure_all_finite=True,
            reset=True,
        )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
            self.feature_names_in_ = X.columns

        if not all([isinstance(x, Hashable) for x in X.values.flat]):
            raise TypeError("argument must be a string, number, or hashable object.")

        return X

    def fit(self, X: pd.DataFrame, y=None):
        """Fit data (`X`) to a causal graph. The method
        calls the `_fit` method, which must be implemented separately in any causal
        discovery algorithm inheriting from `BaseCausalDiscovery`.
        """
        X = self._check_fit_data(X)
        return self._fit(X)


class _ConstraintMixin:
    """
    Base class for all constraint-based causal discovery estimators.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        independencies: Independencies = None,
    ):
        """Fit data (`X`) and independence relations (optional) to a causal graph. The method
        calls the `_fit` method, which must be implemented separately in any causal
        discovery algorithm inheriting from `BaseConstraintCausalDiscovery`.
        """
        X = self._check_fit_data(X)
        return self._fit(X, independencies)

    def _build_skeleton(
        self,
        data,
        independencies=None,
        variant: str = "stable",
        ci_test: Union[str, Callable, None] = None,
        significance_level: float = 0.01,
        max_cond_vars: int = 5,
        expert_knowledge: Optional[ExpertKnowledge] = None,
        enforce_expert_knowledge: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[UndirectedGraph, Dict[Tuple[str, str], Set[str]]]:
        """
        Estimates a graph skeleton (UndirectedGraph) from a set of independencies
        using (the first part of) the PC algorithm.

        The independencies can either be provided as an instance of the
        `Independencies`-class or by passing a decision function that decides any
        conditional independency assertion. Returns a tuple `(skeleton, separating_sets)`.

        If an Independencies-instance is passed, the contained IndependenceAssertions
        have to admit a faithful BN representation. This is the case if
        they are obtained as a set of d-separations of some Bayesian network or
        if the independence assertions are closed under the semi-graphoid axioms.
        Otherwise, the procedure may fail to identify the correct structure.

        Parameters
        ----------
        variant: str (one of "orig", "stable", "parallel")
            The variant of PC algorithm to run.
                "orig": The original PC algorithm. Might not give the same
                        results in different runs but does less independence
                        tests compared to stable.
                "stable": Gives the same result in every run but does needs to
                        do more statistical independence tests.
                "parallel": Parallel version of PC Stable. Can run on multiple
                        cores with the same result on each run.

        ci_test: str or fun
            The statistical test to use for testing conditional independence in
            the dataset. If `str` values should be one of:
                "independence_match": If using this option, an additional parameter
                        `independencies` must be specified.
                "chi_square": Uses the Chi-Square independence test. This works
                        only for discrete datasets.
                "pearsonr": Uses the partial correlation based on pearson
                        correlation coefficient to test independence. This works
                        only for continuous datasets.
                "g_sq": G-test. Works only for discrete datasets.
                "log_likelihood": Log-likelihood test. Works only for discrete dataset.
                "freeman_tuckey": Freeman Tuckey test. Works only for discrete dataset.
                "modified_log_likelihood": Modified Log Likelihood test. Works only for discrete variables.
                "neyman": Neyman test. Works only for discrete variables.
                "cressie_read": Cressie Read test. Works only for discrete variables.

        significance_level: float (default: 0.01)
            The statistical tests use this value to compare with the p-value of
            the test to decide whether the tested variables are independent or
            not. Different tests can treat this parameter differently:
                1. Chi-Square: If p-value > significance_level, it assumes that the
                    independence condition satisfied in the data.
                2. pearsonr: If p-value > significance_level, it assumes that the
                    independence condition satisfied in the data.

        max_cond_vars: int (default: 5)
            The maximum number of variables to condition on while testing
            independence.

        expert_knowledge: pgmpy.estimators.ExpertKnowledge instance
            Expert knowledge to be used with the algorithm. Expert knowledge
            includes required/forbidden edges in the final graph, temporal
            information about the variables etc. Please refer
            pgmpy.estimators.ExpertKnowledge class for more details.

        enforce_expert_knowledge: boolean (default: False)
            If True, the algorithm modifies the search space according to the
            edges specified in expert knowledge object. This implies the following:
                1. For every edge (u, v) specified in `forbidden_edges`, there will
                    be no edge between u and v.
                2. For every edge (u, v) specified in `required_edges`, one of the
                    following would be present in the final model: u -> v, u <-
                    v, or u - v (if CPDAG is returned).

            If False, the algorithm attempts to make the edge orientations as
            specified by expert knowledge after learning the skeleton. This
            implies the following:
                1. For every edge (u, v) specified in `forbidden_edges`, the final
                    graph would have either v <- u or no edge except if u -> v is part
                    of a collider structure in the learned skeleton.
                2. For every edge (u, v) specified in `required_edges`, the final graph
                    would either have u -> v or no edge except if v <- u is part of a
                    collider structure in the learned skeleton.

        n_jobs: int (default: -1)
            The number of jobs to run in parallel.

        show_progress: bool (default: True)
            If True, shows a progress bar while running the algorithm.


        Returns
        -------
        skeleton: UndirectedGraph
            An estimate for the undirected graph skeleton of the BN underlying the data.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes them
            conditionally independent. (needed for edge orientation procedures)

        References
        ----------
        [1] Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [2] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
            Section 3.4.2.1 (page 85), Algorithm 3.3
        """
        # Initialize initial values and structures.
        lim_neighbors = 0
        separating_sets = dict()
        ci_test = ci_registry.get_test(ci_test, data=data)

        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(data.columns)

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Working for n conditional variables: 0")

        variables = list(data.columns.values)

        # Step 1: Initialize a fully connected undirected graph
        graph = nx.complete_graph(n=variables, create_using=nx.Graph)
        temporal_ordering = expert_knowledge.temporal_ordering
        if enforce_expert_knowledge:
            graph.remove_edges_from(expert_knowledge.forbidden_edges)

        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in variables]
        ):
            # Step 2: Iterate over the edges and find a conditioning set of
            # size `lim_neighbors` which makes u and v independent.
            if variant == "orig":
                for u, v in graph.edges():
                    if (enforce_expert_knowledge is False) or (
                        (u, v) not in expert_knowledge.required_edges
                    ):
                        for separating_set in self._get_potential_sepsets(
                            u, v, temporal_ordering, graph, lim_neighbors
                        ):
                            # If a conditioning set exists remove the edge, store the separating set
                            # and move on to finding conditioning set for next edge.
                            if ci_test(
                                u,
                                v,
                                separating_set,
                                data=data,
                                independencies=independencies,
                                significance_level=significance_level,
                                **kwargs,
                            ):
                                separating_sets[frozenset((u, v))] = separating_set
                                graph.remove_edge(u, v)
                                break

            elif variant == "stable":
                # In case of stable, precompute neighbors as this is the stable algorithm.
                for u, v in graph.edges():
                    if (enforce_expert_knowledge is False) or (
                        (u, v) not in expert_knowledge.required_edges
                    ):
                        for separating_set in self._get_potential_sepsets(
                            u, v, temporal_ordering, graph, lim_neighbors
                        ):
                            # If a conditioning set exists remove the edge, store the
                            # separating set and move on to finding conditioning set for next edge.
                            if ci_test(
                                u,
                                v,
                                separating_set,
                                data=data,
                                independencies=independencies,
                                significance_level=significance_level,
                                **kwargs,
                            ):
                                separating_sets[frozenset((u, v))] = separating_set
                                graph.remove_edge(u, v)
                                break

            elif variant == "parallel":

                def _parallel_fun(u, v):
                    for separating_set in self._get_potential_sepsets(
                        u, v, temporal_ordering, graph, lim_neighbors
                    ):
                        if ci_test(
                            u,
                            v,
                            separating_set,
                            data=data,
                            independencies=independencies,
                            significance_level=significance_level,
                            **kwargs,
                        ):
                            return (u, v), separating_set

                results = Parallel(n_jobs=n_jobs)(
                    delayed(_parallel_fun)(u, v)
                    for (u, v) in graph.edges()
                    if (enforce_expert_knowledge is False)
                    or ((u, v) not in expert_knowledge.required_edges)
                )
                for result in results:
                    if result is not None:
                        (u, v), sep_set = result
                        graph.remove_edge(u, v)
                        separating_sets[frozenset((u, v))] = sep_set

            else:
                raise ValueError(
                    f"variant must be one of (orig, stable, parallel). Got: {variant}"
                )

            # Step 3: After iterating over all the edges, expand the search space by increasing the size
            #         of conditioning set by 1.
            if lim_neighbors >= max_cond_vars:
                logger.info(
                    "Reached maximum number of allowed conditional variables. Exiting"
                )
                break
            lim_neighbors += 1

            if show_progress and config.SHOW_PROGRESS:
                pbar.update(1)
                pbar.set_description(
                    f"Working for n conditional variables: {lim_neighbors}"
                )

        if show_progress and config.SHOW_PROGRESS:
            pbar.update(max_cond_vars - lim_neighbors)
            pbar.close()

        return graph, separating_sets

    @staticmethod
    def _get_potential_sepsets(
        u: Hashable,
        v: Hashable,
        temporal_ordering: Dict[Hashable, int],
        graph: UndirectedGraph,
        lim_neighbors: int,
    ) -> Collection[Tuple]:
        """
        Return the temporally consistent superset of separating set of `u`, `v`.

        The temporal order (if specified) of the superset can only be smaller
        ("earlier") than a particular node. The neighbors of `u` satisfying
        this condition are returned.

        Parameters
        ----------
        u: variable
            The node whose neighbors are being considered for separating set.

        v: variable
            The node along with u whose separating set is being calculated.

        temporal_ordering: dict
            The temporal ordering of variables according to prior knowledge.

        graph: UndirectedGraph
            The graph where separating sets are being calculated for the edges.

        lim_neighbors: int
            The maximum number of neighbours (conditioning variables) for u, v.

        Returns
        --------
        separating_set: set
            Set containing the superset of separating set of u, v.
        """
        separating_set_u = set(graph.neighbors(u))
        separating_set_v = set(graph.neighbors(v))
        separating_set_u.discard(v)
        separating_set_v.discard(u)

        if temporal_ordering != dict():
            max_order = min(temporal_ordering[u], temporal_ordering[u])
            for neigh in list(separating_set_u):
                if temporal_ordering[neigh] > max_order:
                    separating_set_u.discard(neigh)

            for neigh in list(separating_set_v):
                if temporal_ordering[neigh] > max_order:
                    separating_set_v.discard(neigh)

        return chain(
            combinations(separating_set_u, lim_neighbors),
            combinations(separating_set_v, lim_neighbors),
        )
