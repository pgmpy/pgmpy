#!/usr/bin/env python

from itertools import chain, combinations, permutations
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

import pandas as pd
import networkx as nx
from tqdm import tqdm

from pgmpy import config
from pgmpy.estimators.BaseConstraintEstimator import BaseConstraintEstimator
from pgmpy.estimators.PC import PC
from pgmpy.base import UndirectedGraph
from pgmpy.independencies import Independencies
from pgmpy.global_vars import logger
from pgmpy.estimators.CITests import get_callable_ci_test
from pgmpy.base.ancestral.base import AncestralGraph


class FCI(BaseConstraintEstimator, AncestralGraph):
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
        """
        Initializes the FCI estimator.

        Parameters
        ----------
        edge_types: Tuple[str, str]
            Tuple of edge types, typically ('circle', 'tail') or ('circle', 'arrowhead').
        data: Optional[pd.DataFrame]
            Data for structure estimation.
        independencies: Optional[Independencies]
            Pre-defined independencies.
        **kwargs:
            Additional keyword arguments.
        """
        super(BaseConstraintEstimator, self).__init__(
            data=data, independencies=independencies, **kwargs
        )
        self.edge_types = edge_types
        self.graph = None
        self._separating_sets = None

    def build_skeleton(
        self,
        variant: str = "stable",
        ci_test: Union[str, Callable, None] = None,
        significance_level: float = 0.01,
        max_cond_vars: int = 5,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[UndirectedGraph, Dict[Tuple[str, str], Set[str]]]:
        """
        Builds the skeleton of the graph.

        This method overrides the parent's `build_skeleton` to implement
        the specific logic for FCI, but it can still leverage the shared
        `_get_potential_sepsets` method from the base class.

        Parameters
        ----------
        variant: str (default: "stable")
            The variant of the algorithm.
        ci_test: Union[str, Callable, None] (default: None)
            The conditional independence test.
        significance_level: float (default: 0.01)
            The significance level for the CI test.
        max_cond_vars: int (default: 5)
            The maximum size of the conditioning set.
        show_progress: bool (default: True)
            Whether to show a progress bar.
        **kwargs:
            Additional keyword arguments for the CI test.

        Returns
        -------
        Tuple[UndirectedGraph, Dict[Tuple[str, str], Set[str]]]
            The graph skeleton and the separating sets.
        """

        # Initialize initial values and structures
        lim_neighbors = 0
        separating_sets = dict()
        ci_test = get_callable_ci_test(ci_test, full=True, data=self.data)

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Working for n conditional variables")

        # Step1: Initialize the undirected graph
        graph = nx.complete_graph(n=self.variables, create_using=nx.Graph)
        # currently skipping the temporal ordering due to lack of expert knowledge
        temporal_ordering = (
            {}
        )  # FCI doesn't use temporal ordering in the same way as PC

        # Exit Condition: (AS PER THE PC algorithm)
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in self.variables]
        ):
            # Step2: iterate over all the edges and find a conditioning set
            # of size `lim_neighbors` which makes u and v independent

            # Currently including only a single variant
            if variant == "orig":
                for u, v in graph.edges():
                    for separating_set in self._get_potential_sepsets(
                        u, v, temporal_ordering, graph, lim_neighbors=lim_neighbors
                    ):
                        # If a conditioning set exists, remove the edge,
                        # store the separating set and move on to finding
                        # the conditioning set for the next edge.
                        if ci_test(
                            u,
                            v,
                            separating_set,
                            data=self.data,
                            independencies=self.independencies,
                            significance_level=significance_level,
                            **kwargs,
                        ):
                            separating_sets[frozenset((u, v))] = separating_set
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

    def _orient_colliders(self):
        """
        Orients unshielded triples as colliders (Rule R0).

        An unshielded triple (a, c, b) is oriented as a collider a*->c<-*b if
        and only if c is not in the separating set of a and b.
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

            # Rule 1: If a*->b o-*c, and a and c are not adjacent, then orient a*->b->c.
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

            # Rule 2: If α->β*->γ, α*-o γ, and α and γ are adjacent, then orient α*-o γ as α*->γ.
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

            # Rule 3: If α*->β<-*γ, α and γ are not adjacent, and θ*-o β, then orient θ*-o β as θ*->β.
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
