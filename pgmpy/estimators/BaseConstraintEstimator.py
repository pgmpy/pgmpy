#!/usr/bin/env python

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
from tqdm import tqdm

from pgmpy import config
from pgmpy.base import UndirectedGraph
from pgmpy.estimators import ExpertKnowledge, StructureEstimator
from pgmpy.estimators.CITests import get_callable_ci_test
from pgmpy.global_vars import logger
from pgmpy.independencies import Independencies


class BaseConstraintEstimator(StructureEstimator):
    """
    Base class for constraint-based causal discovery algorithms like PC and
    FCI.

    This class provides common methods for building the skeleton of a graph,
    handling separating sets, and managing the core logic of the
    skeleton-building phase, which is shared by many constraint-based
    algorithms.

    Attributes
    ----------
    data: Optional[pd.DataFrame]
        The data from which to learn the graph structure.
    independencies: Optional[Independencies]
        A pre-defined set of independencies to use instead of a dataset.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        independencies: Optional[Independencies] = None,
        **kwargs,
    ) -> None:
        """
        Identify conditional independencies in the given dataset, as used in
        constraint-based causal discovery algorithms.

        Constraint-based methods rely on statistical tests to detect whether
        two variables are conditionally independent given a conditioning set.
        These `independencies` form the backbone of causal graph structure
        learning by progressively removing the edges that are unsupported
        by the data.
        """
        super().__init__(data=data, independencies=independencies, **kwargs)

    def _get_potential_sepsets(
        self,
        u: Hashable,
        v: Hashable,
        temporal_ordering: Dict[Hashable, int],
        graph: UndirectedGraph,
        lim_neighbors: int,
    ) -> Collection[Tuple]:
        """
        Generates potential separating sets for an edge (u, v) based on the
        current graph and temporal ordering.

        This method identifies neighbors of u and v (excluding each other) and
        then generates combinations of these neighbors of a specified size
        (`lim_neighbors`). Temporal ordering is used to filter out neighbors
        that occur after both u and v.

        Parameters
        ----------
        u: Hashable
            The first node of the edge.
        v: Hashable
            The second node of the edge.
        temporal_ordering: Dict[Hashable, int]
            A dictionary mapping variables to their temporal order.
        graph: UndirectedGraph
            The current state of the undirected graph.
        lim_neighbors: int
            The size of the conditioning sets to be generated.

        Returns
        -------
        Collection[Tuple]
            An iterable of tuples, where each tuple is a potential separating
            set.
        """
        separating_set_u = set(graph.neighbors(u))
        separating_set_v = set(graph.neighbors(v))
        separating_set_u.discard(v)
        separating_set_v.discard(u)

        # If no temporal ordering provided, skip filtering
        if not temporal_ordering:
            return chain(
                combinations(separating_set_u, lim_neighbors),
                combinations(separating_set_v, lim_neighbors),
            )

        try:
            max_order = min(temporal_ordering[u], temporal_ordering[v])
        except KeyError as e:
            raise KeyError(
                f"Node {e.args[0]} not found in temporal_ordering."
            ) from None

        for neigh in list(separating_set_u):
            if neigh not in temporal_ordering:
                raise KeyError(f"{neigh} not found in temporal_ordering.")
            if temporal_ordering[neigh] > max_order:
                separating_set_u.discard(neigh)

        for neigh in list(separating_set_v):
            if neigh not in temporal_ordering:
                raise KeyError(f"{neigh} not found in temporal_ordering.")
            if temporal_ordering[neigh] > max_order:
                separating_set_v.discard(neigh)

        return chain(
            combinations(separating_set_u, lim_neighbors),
            combinations(separating_set_v, lim_neighbors),
        )

    def build_skeleton(
        self,
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
        Estimates a graph skeleton (UndirectedGraph) from a set of
        independencies using (the first part of) the PC algorithm. The
        independencies can either be provided as an instance of the
        `Independencies`-class or by passing a decision function that decides
        any conditional independency assertion. Returns a tuple `(skeleton,
        separating_sets)`.

        If an Independencies-instance is passed, the contained
        IndependenceAssertions have to admit a faithful BN representation.
        This is the case if they are obtained as a set of d-separations of
        some Bayesian network or if the independence assertions are closed
        under the semi-graphoid axioms. Otherwise, the procedure may fail to
        identify the correct structure.

        References
        ----------
        [1] Neapolitan, Learning Bayesian Networks, Section 10.1.2,
            Algorithm 10.2 (page 550)
            http://www.cs.technion.ac.il/~dang/books/
            Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [2] Koller & Friedman, Probabilistic Graphical Models - Principles
            and Techniques, 2009, Section 3.4.2.1 (page 85), Algorithm 3.3
        """
        lim_neighbors = 0
        separating_sets = dict()
        ci_test = get_callable_ci_test(ci_test, full=True, data=None)

        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(self.data.columns)

        # Default temporal ordering if missing/empty
        temporal_ordering = expert_knowledge.temporal_ordering
        if not temporal_ordering:
            temporal_ordering = {var: 0 for var in self.variables}

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Working for n conditional variables: 0")

        # Step 1: Initialize fully connected graph
        graph = nx.complete_graph(n=self.variables, create_using=nx.Graph)
        if enforce_expert_knowledge:
            graph.remove_edges_from(expert_knowledge.forbidden_edges)

        neighbor_counts = (len(graph.neighbors(v)) for v in self.variables)
        while not all(count < lim_neighbors for count in neighbor_counts):

            if variant == "orig":
                for u, v in list(graph.edges()):
                    if (
                        enforce_expert_knowledge is False
                        or (u, v) not in expert_knowledge.required_edges
                    ):
                        for sep_set in self._get_potential_sepsets(
                            u, v, temporal_ordering, graph, lim_neighbors
                        ):
                            if ci_test(
                                u,
                                v,
                                sep_set,
                                data=self.data,
                                independencies=self.independencies,
                                significance_level=significance_level,
                                **kwargs,
                            ):
                                separating_sets[frozenset((u, v))] = sep_set
                                graph.remove_edge(u, v)
                                break

            elif variant == "stable":
                for u, v in list(graph.edges()):
                    if (
                        enforce_expert_knowledge is False
                        or (u, v) not in expert_knowledge.required_edges
                    ):
                        for separating_set in self._get_potential_sepsets(
                            u, v, temporal_ordering, graph, lim_neighbors
                        ):
                            if ci_test(
                                u,
                                v,
                                sep_set,
                                data=self.data,
                                independencies=self.independencies,
                                significance_level=significance_level,
                                **kwargs,
                            ):
                                separating_sets[frozenset((u, v))] = sep_set
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
                            sep_set,
                            data=self.data,
                            independencies=self.independencies,
                            significance_level=significance_level,
                            **kwargs,
                        ):
                            return (u, v), sep_set

                results = Parallel(n_jobs=n_jobs)(
                    delayed(_parallel_fun)(u, v)
                    for (u, v) in list(graph.edges())
                    if (
                        enforce_expert_knowledge is False
                        or (u, v) not in expert_knowledge.required_edges
                    )
                )
                for result in results:
                    if result is not None:
                        (u, v), sep_set = result
                        graph.remove_edge(u, v)
                        separating_sets[frozenset((u, v))] = sep_set

            else:
                raise ValueError(
                    "variant must be one of (orig, stable, parallel). "
                    f"Got: {variant}"
                )

            if lim_neighbors >= max_cond_vars:
                logger.info(
                    "Reached maximum number of allowed conditional "
                    "variables. Exiting"
                )
                break
            lim_neighbors += 1

            if show_progress and config.SHOW_PROGRESS:
                pbar.update(1)
                pbar.set_description(
                    f"Working for n conditional variables: " f"{lim_neighbors}"
                )

        if show_progress and config.SHOW_PROGRESS:
            pbar.update(max_cond_vars - lim_neighbors)
            pbar.close()
        return graph, separating_sets
