#!/usr/bin/env python

from itertools import chain, combinations, permutations
from tqdm import tqdm

import networkx as nx
import pandas as pd
from typing import (
    Callable,
    Collection,
    Dict,
    Hashable,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from pgmpy import config
from pgmpy.estimators.PC import PC
from pgmpy.estimators import StructureEstimator
from pgmpy.base import DAG, PDAG, UndirectedGraph
from pgmpy.independencies import Independencies
from pgmpy.global_vars import logger
from pgmpy.estimators.CITests import get_callable_ci_test
from pgmpy.base.ancestral.base import AncestralGraph


class FCI(StructureEstimator, AncestralGraph):
    """
    An implementation of the FCI (Fast Causal Inference) algorithm.

    The FCI algorithm is a causal discovery method that extends the PC algorithm
    to handle latent confounding variables and selection bias.
    It outputs a Partial Ancestral Graph (PAG) which uses a combination of
    circles(o), tails(--), and arrowheads(>) to represent the
    causal relationships.
    """

    def __init__(
        self,
        edge_types: Tuple[str, str],
        data: Optional[pd.DataFrame] = None,
        independencies: Optional[Independencies] = None,
        **kwargs,
    ) -> None:
        super(PC, self).__init__(data, independencies, edge_types, **kwargs)

    def build_skeleton(
        self,
        variant: str = "stable",
        ci_test: Union[str, Callable, None] = None,
        significance_level: float = 0.01,
        max_cond_vars: int = 5,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[UndirectedGraph, Dict[Tuple[str, str], Set[str]]]:

        # Initilize initial values and structures
        lim_neighbors = 0
        separating_sets = dict()
        ci_test = get_callable_ci_test(ci_test, full=True, data=None)

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Working for n conditional variables")

        # Step1: Initialize the uindirected graph
        graph = nx.complete_graph(n=self.variables, create_using=nx.Graph)
        # currently skipping the temporal ordering due to lack of expert knowledge

        # Exit Condition: (AS PER THE PC algorithm)
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in self.variables]
        ):
            # Step2: iterate over all the edges and find a conditioning set
            # of size `lim_neighbors` which makes u and v independent

            # Currently including only a single variant
            if variant == "orig":
                for u, v in graph.edges():
                    for separating_set in PC._get_potential_sepsets(
                        u, v, graph, lim_neighbors=lim_neighbors
                    ):
                        # If a conditioning set exists, remove the edge,
                        # store the separating set and move on to finding
                        # the conditioning set for the next edge.
                        if ci_test(
                            u,
                            v,
                            separating_set,
                            data=self.data,
                            Independencies=self.independencies,
                            significance_level=significance_level,
                            **kwargs,
                        ):
                            separating_set[frozenset((u, v))] = separating_set
                            graph.remove_edge(u, v)
                            break

            # step3 : After iterating over al the edges, expand the search space,
            # by increasing the size of the conditioning set
            if lim_neighbors >= max_cond_vars:
                logger.info("Reached maximum number of allowed variables. Exiting")
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
        Return the temporally consistent superset of separating set of u, v.
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

    def _orient_colliders(self):
        """
        Orients unshielded triples as colliders (Rule R0).

        An unshielded triple (a, c, b) is oriented as a collider a*->c<-*b if
        [cite_start]and only if c is not in the separating set of a and b[cite: 40].
        """
        for a in self.graph.nodes():
            for b in self.graph.neighbors(a):
                for c in self.graph.neighbors(b):
                    if a == c or b == a:
                        continue
                    # Check for unshielded triple
                    if not self.graph.has_edge(a, c):
                        if (
                            c not in self._separating_sets[frozenset((a, b))]
                            and a not in self._separating_sets[frozenset((c, b))]
                        ):
                            # a*->b<-*c
                            self.graph.add_edge(a, b, mark="arrowhead")
                            self.graph.add_edge(c, b, mark="arrowhead")

    def _orient_edges(self):
        """
        Applies the 10 orientation rules (R1-R10) iteratively.
        """
        changed = True
        while changed:
            changed = False

            # [cite_start]Rule 1: If a*->b o-*c, and a and c are not adjacent, then orient a*->b->c[cite: 41].
            for b in self.graph.nodes():
                for a in self.graph.predecessors(b):
                    for c in self.graph.successors(b):
                        if a == c:
                            continue
                        if (
                            self.graph.has_edge(a, b)
                            and self.graph.get_edge_data(a, b).get("mark")
                            == "arrowhead"
                            and self.graph.has_edge(b, c)
                            and self.graph.get_edge_data(b, c).get("mark") == "circle"
                            and not self.graph.has_edge(a, c)
                            and not self.graph.has_edge(c, a)
                        ):
                            # Orient b o-* c as b -> c
                            self.graph.add_edge(b, c, mark="arrowhead")
                            self.graph.add_edge(c, b, mark="tail")
                            changed = True

            # [cite_start]Rule 2: If α->β*->γ, α*-o γ, and α and γ are adjacent, then orient α*-o γ as α*->γ[cite: 42].
            for a in self.graph.nodes():
                for c in self.graph.neighbors(a):
                    if (
                        self.has_circle_at(c, a)
                        and self.has_tail_at(a, c)
                        and not self.graph.has_edge(a, c)
                    ):  # α*-o γ
                        for b in self.graph.nodes():
                            if b == a or b == c:
                                continue
                            if self.is_directed(a, b) and self.has_arrowhead_at(
                                b, c
                            ):  # α->β*->γ
                                self.add_edge(a, c, "tail", "arrowhead")
                                changed = True

            # [cite_start]Rule 3: If α*->β<-*γ, α and γ are not adjacent, and θ*-o β, then orient θ*-o β as θ*->β[cite: 43].
            for b in self.graph.nodes():
                for a, c in permutations(self.graph.predecessors(b), 2):
                    if (
                        not self.graph.has_edge(a, c)
                        and self.graph.get_edge_data(a, b).get("mark") == "arrowhead"
                        and self.graph.get_edge_data(c, b).get("mark") == "arrowhead"
                    ):
                        for d in self.graph.predecessors(b):
                            if (
                                d != a
                                and d != c
                                and self.graph.get_edge_data(d, b).get("mark")
                                == "circle"
                            ):
                                # Orient d*-o b as d*->b
                                self.graph.add_edge(d, b, mark="arrowhead")
                                changed = True
        return self.graph
