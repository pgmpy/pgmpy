#!/usr/bin/env python

from itertools import permutations, combinations, chain
from typing import (
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Hashable,
    Optional,
    Set,
    Tuple,
    Union,
)

import pandas as pd
import networkx as nx
from tqdm import tqdm

from pgmpy import config
from pgmpy.estimators import StructureEstimator, ExpertKnowledge
from pgmpy.estimators.CITests import get_callable_ci_test
from pgmpy.independencies import Independencies
from pgmpy.base import UndirectedGraph
from pgmpy.base.DAG import DAG
from pgmpy.global_vars import logger


class BaseConstraintEstimator(StructureEstimator, DAG):
    """
    Base class for constraint-based causal discovery algorithms like
    PC and FCI.

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
    variables: list
        The list of variables (columns) in the data.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        independencies: Optional[Independencies] = None,
        **kwargs,
    ) -> None:
        super(BaseConstraintEstimator, self).__init__(
            data=data, independencies=independencies, **kwargs
        )

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
            An iterable of tuples, where each tuple is a potential separating set.
        """
        separating_set_u = set(graph.neighbors(u))
        separating_set_v = set(graph.neighbors(v))
        separating_set_u.discard(v)
        separating_set_v.discard(u)

        if temporal_ordering != {}:
            max_order = min(
                temporal_ordering.get(u, float("inf")),
                temporal_ordering.get(v, float("inf")),
            )

            # Filter neighbors based on temporal ordering
            for neigh in list(separating_set_u):
                if temporal_ordering.get(neigh, float("inf")) > max_order:
                    separating_set_u.discard(neigh)

            for neigh in list(separating_set_v):
                if temporal_ordering.get(neigh, float("inf")) > max_order:
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
    ) -> Tuple[UndirectedGraph, Dict[FrozenSet, Tuple]]:
        """
        Constructs the skeleton of the graph by iteratively removing edges
        that are rendered independent by a conditioning set.

        This is the core skeleton-building method for constraint-based
        estimators. It starts with a complete graph and prunes edges based on
        conditional independence tests.

        Parameters
        ----------
        variant: str (one of "orig", "stable", "parallel")
            The variant of the algorithm to run.
        ci_test: str or Callable
            The statistical test to use for conditional independence.
        significance_level: float (default: 0.01)
            The significance level for the statistical tests.
        max_cond_vars: int (default: 5)
            The maximum size of the conditioning set to test.
        expert_knowledge: Optional[ExpertKnowledge]
            Expert knowledge to be used with the algorithm.
        enforce_expert_knowledge: bool (default: False)
            If True, the algorithm strictly enforces expert knowledge
            (e.g., removing forbidden edges from the initial graph).
        n_jobs: int (default: -1)
            The number of parallel jobs to run (for "parallel" variant).
        show_progress: bool (default: True)
            If True, displays a progress bar.

        Returns
        -------
        Tuple[UndirectedGraph, Dict[FrozenSet, Tuple]]
            A tuple containing the estimated skeleton (UndirectedGraph) and a
            dictionary of separating sets for each removed edge.
        """
        # Step 0: Initialize values and structures.
        lim_neighbors = 0
        separating_sets = dict()
        ci_test = get_callable_ci_test(
            ci_test, full=True, data=self.data, independencies=self.independencies
        )

        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(self.data.columns)

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Working for n conditional variables: 0")

        # Step 1: Initialize a fully connected undirected graph.
        graph = nx.complete_graph(n=self.variables, create_using=nx.Graph)
        temporal_ordering = expert_knowledge.temporal_ordering
        if enforce_expert_knowledge:
            graph.remove_edges_from(expert_knowledge.forbidden_edges)

        # Step 2: Iteratively prune edges.
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in self.variables]
        ):
            edges_to_remove = []
            current_edges = list(graph.edges())

            for u, v in current_edges:
                # Check if the edge should be kept due to expert knowledge
                if (
                    u,
                    v,
                ) in expert_knowledge.required_edges and enforce_expert_knowledge:
                    continue

                for separating_set in self._get_potential_sepsets(
                    u, v, temporal_ordering, graph, lim_neighbors
                ):
                    if ci_test(
                        u,
                        v,
                        separating_set,
                        data=self.data,
                        independencies=self.independencies,
                        significance_level=significance_level,
                        **kwargs,
                    ):
                        edges_to_remove.append((u, v, separating_set))
                        break

            for u, v, separating_set in edges_to_remove:
                separating_sets[frozenset((u, v))] = separating_set
                if graph.has_edge(u, v):
                    graph.remove_edge(u, v)

            # Step 3: Expand the search space.
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
            pbar.close()

        return graph, separating_sets
