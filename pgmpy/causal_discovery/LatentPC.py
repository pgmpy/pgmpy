from collections import deque
from joblib import Parallel, delayed
from collections.abc import Callable, Generator, Hashable
from itertools import combinations, permutations
from collections.abc import Callable, Hashable
from itertools import combinations, chain
from tqdm.auto import tqdm
from typing import List, Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.utils.validation import check_is_fitted, validate_data


from pgmpy import config, logger
from pgmpy.base import PDAG, MAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.causal_discovery._base import BaseCausalDiscovery, _ConstraintMixin
from pgmpy.ci_tests import get_ci_test, IndependenceMatch


class LatentPC(_ConstraintMixin):
    """
    The Ψ-PC algorithm for causal discovery / structure learning.

    This class implements the Ψ-PC algorithm [1] for causal discovery. Ψ-PC is an
    extension of the PC algorithm that accommodates (unknown) latent variables. Given
    a set of tabular datasets, the Ψ-PC algorithm estimates the causal structure among the
    variables in the data as a Ψ-Partially Directed Acyclic Graph (Ψ-PDAG).

    The algorithm uses data from a combination of observational and experimental data (soft
    interventions), with the interventional targets being unknown.
    The algorithm works by combining conditional independency tests and mechanism shift tests
    and estimates a DAG pattern that satisfies the identified dependencies.
    Ψ-PC can be used for structural learning in non-Markovian systems (i.e., when latent variables affect
    more than one observable).

    Parameters
    ----------
    ci_test : str or callable, default=None
        The conditional independence (CI) test to use for finding (conditional) independences in the data. This can be
        any of the CI test implemented in :mod:`pgmpy.ci_tests` or a custom function that follows the
        signature of the built-in CI tests.

        If None, the appropriate CI test will be chosen based on the data type.

    return_type : str, default="pdag"
        The type of structure to return. Can be one of: `pdag`, `cpdag`, `dag`.

        - If `return_type=pdag` or `return_type=cpdag`: a partially directed structure is returned.
        - If `return_type=dag`, a fully directed structure is returned. This DAG is one of the possible orientations of
          the PDAG learned by the PC algorithm.

    significance_level : float, default=0.01
        The p-value threshold to use for the statistical independence tests. If the p-value of a test is greater than
        `significance_level`, then the variables are considered independent.

    max_cond_vars : int, default=5
        The maximum number conditional variables to consider while performing conditional independence tests.

    orient_rule : str or None, default=None
        The rule for orienting colliders (v-structures) when there is a conflict.

        - ``None``: The first orientation is kept, later conflicting orientations are ignored.
        - ``"pvalue"``: For each candidate collider at ``Z``, CI tests are run over all subsets ``S`` of the neighbors.
          ``Z`` is considered a collider if the maximum p-value over subsets not containing ``Z`` exceeds the maximum
          over subsets containing ``Z``. Candidate colliders are then sorted by strength (highest p-value first) to
          resolve conflicts.
        - ``"effect"``: Same as ``"pvalue"`` but uses effect sizes instead of p-values for testing colliders and
          resolving conflicts.

    expert_knowledge : :class:`pgmpy.causal_discovery.ExpertKnowledge`, optional
        Expert knowledge to be used in the causal graph construction. This needs to be an instance of
        :class:`pgmpy.causal_discovery.ExpertKnowledge`. Users can specify knowledge in the form of
        required/forbidden edges, temporal information, or restrict the search space. The knowledge is
        always applied as hard directional constraints after the skeleton is learned (see the note above).

    n_jobs : int, default=-1
        The number of jobs to run in parallel. This is only used when `variant="parallel"`.

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    Attributes
    ----------
    causal_graph_ : :class:`~pgmpy.base.DAG` or :class: `~pgmpy.base.PDAG`
        The learned causal graph.

        - If `return_type="pdag"`, this will be a PDAG instance.
        - If `return_type="dag"`, this will be a DAG instance.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph, i.e. `causal_graph_`.

    skeleton_ : :class:`~pgmpy.base.UndirectedGraph`
        An estimate for the undirected graph skeleton of the DAG underlying the data.

    separating_sets_ : dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes them
            conditionally independent. (needed for edge orientation procedures)

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    """

    def __init__(
        self,
        variant: str = "parallel",
        ci_test: str | Callable | None = None,
        return_type: str = "pdag",
        significance_level: float = 0.01,
        max_cond_vars: int = 5,
        orient_rule: str | None = None,
        expert_knowledge: ExpertKnowledge | None = None,
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
        self.variant = variant
        self.ci_test = ci_test
        self.return_type = return_type
        self.significance_level = significance_level
        self.max_cond_vars = max_cond_vars
        self.orient_rule = orient_rule
        self.expert_knowledge = expert_knowledge
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self._ci_test_mechanisms = {}

    def _fit(self, datasets: list[pd.DataFrame], independencies=None):
        """
        The fitting procedure for the PC algorithm.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The data to learn the causal structure from. If a numpy array is
            passed, then the column names would be integers from 0 to n_features-1.

        Returns
        -------
        self : pgmpy.causal_discovery.PC
            Returns the instance with the fitted attributes.
        """

        # CI test
        self.ci_test_ = get_ci_test(test=self.ci_test, data=datasets[0])

        # Check if expert knowledge was specified
        if self.expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()
        else:
            # Clone so the fitted (`*_`) attributes land on a fresh copy, not the user's object.
            expert_knowledge = clone(self.expert_knowledge)

        # Resolve the expert knowledge into its fitted (`*_`) attributes.
        X = datasets[0]
        expert_knowledge.fit(X)

        # Step 1: Build the skeleton

        lim_neighbors = 0
        separating_sets = dict()

        # if independencies is not None:
        #    ci_test = IndependenceMatch(independencies=independencies)
        # else:
        #    ci_test = get_ci_test(test=self.ci_test, data=datasets[0])

        if self.show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=self.max_cond_vars)
            pbar.set_description("Working for n conditional variables: 0")

        variables = list(X.columns.values)

        # Step 1: Initialize a fully connected undirected graph
        f_nodes = []
        distribution_map = {}
        k = 1
        for i in range(1, len(datasets) + 1):
            for j in range(i + 1, len(datasets) + 1):
                if f"f_{k}" in variables:
                    raise ValueError(
                        f"Observed variable f_{k} conflicts with a mechanism-node name."
                    )
                distribution_map[f"f_{k}"] = (i, j)
                f_nodes.append(f"f_{k}")
                k += 1

        all_variables = variables + f_nodes
        self.latent_nodes = set(f_nodes)
        graph = nx.complete_graph(n=all_variables, create_using=nx.Graph)
        if expert_knowledge is None:
            temporal_ordering, required_edges, forbidden_edges = {}, set(), set()
        else:
            temporal_ordering = expert_knowledge.temporal_ordering_
            required_edges = expert_knowledge.required_edges_
            forbidden_edges = expert_knowledge.forbidden_edges_

        # Remove edges that are forbidden in both directions. Directed forbidden are enforced as orientations after the
        # skeleton is learned. Also remove edges between mechanism nodes.
        graph.remove_edges_from(
            [(u, v) for (u, v) in forbidden_edges if (v, u) in forbidden_edges]
        )
        for idx, f1 in enumerate(f_nodes):
            for f2 in f_nodes[idx + 1 :]:
                graph.remove_edge(f1, f2)
                separating_sets[frozenset((f1, f2))] = set()

        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in variables]
        ):
            # Step 2.1: Iterate over the edges and find a conditioning set of
            # size `lim_neighbors` which makes u and v independent.

            curr_edges = list(graph.edges())
            for u, v in curr_edges:
                edges_to_remove = []
                if u in f_nodes and v in f_nodes:
                    sep_flag, sep_set = True, set()

                elif (u, v) not in required_edges:
                    sep_flag, sep_set = self.latent_separating_set(
                        datasets,
                        u,
                        v,
                        variables,
                        f_nodes,
                        distribution_map,
                        lim_neighbors,
                    )

                separating_sets[frozenset((u, v))] = sep_set

                if sep_flag is True:
                    edges_to_remove.append((u, v))

                graph.remove_edges_from(edges_to_remove)

            # Step 2.2: After iterating over all the edges, expand the search space by increasing the size
            #         of conditioning set by 1.
            if lim_neighbors >= self.max_cond_vars:
                logger.info(
                    f"Reached the maximum number of conditional variables ({self.max_cond_vars}). Exiting."
                )
                break

            lim_neighbors += 1
            if self.show_progress and config.SHOW_PROGRESS:
                pbar.update(1)
                pbar.set_description(
                    f"Working for n conditional variables: {lim_neighbors}"
                )

        if self.show_progress and config.SHOW_PROGRESS:
            pbar.update(self.max_cond_vars - lim_neighbors)
            pbar.close()

        # Step 3: Use separating sets to orient colliders
        self.skeleton_, self.separating_sets_ = graph, separating_sets
        pdag = self._orient_colliders(
            temporal_ordering=expert_knowledge.temporal_ordering_,
        )

        # Step 4: Apply orientation rules and expert knowledge as hard directional constraints.
        curr_edges = list(pdag.edges)
        for u, v, _ in curr_edges:
            if u in f_nodes:
                pdag.orient_undirected_edge(u, v, inplace=True)

            elif v in f_nodes:
                pdag.orient_undirected_edge(v, u, inplace=True)

        pdag = pdag.apply_meeks_rules(apply_r4=False)
        pdag.add_nodes_from(set(X.columns) - set(pdag.nodes()))

        self.causal_graph_ = pdag
        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )
        return self

    def latent_separating_set(
        self,
        datasets: List[pd.DataFrame],
        x: Any,
        y: Any,
        graph_nodes: Iterable,
        latent_nodes: Iterable,
        distribution_map: dict,
        limit_neighbors: int = 5,
    ):

        graph_nodes = set(graph_nodes)
        latent_nodes = set(latent_nodes)

        if x not in latent_nodes and y in latent_nodes:
            x, y = y, x  # make x the F-node (latent node)

        if x not in latent_nodes and y not in latent_nodes:
            # choose any arbitrary distribution
            ci_test = getattr(self, "ci_test_", None)
            if ci_test is None:
                ci_test = get_ci_test(test=self.ci_test, data=datasets[0])
                self.ci_test_ = ci_test

            conditioning_set = list(graph_nodes - {x, y})
            conditioning_set = list(
                chain.from_iterable(
                    combinations(conditioning_set, r)
                    for r in range(limit_neighbors + 1)
                )
            )
            for nodes in conditioning_set:
                if ci_test.is_independent(
                    x, y, nodes, significance_level=self.significance_level
                ):
                    return True, set(nodes) | latent_nodes

        else:
            i, j = distribution_map[x]
            conditioning_set = list(graph_nodes - {y})
            conditioning_set = list(
                chain.from_iterable(
                    combinations(conditioning_set, r)
                    for r in range(limit_neighbors + 1)
                )
            )

            if x not in self._ci_test_mechanisms:

                d_i = datasets[i - 1].copy()
                d_j = datasets[j - 1].copy()

                d_i["_mechanism"] = 0
                d_j["_mechanism"] = 1

                data = pd.concat(
                    [d_i, d_j],
                    ignore_index=True,
                )
                data["_mechanism"] = data["_mechanism"].astype(pd.CategoricalDtype())

                ci_test = get_ci_test(
                    self.ci_test,
                    data=data,
                )
                self._ci_test_mechanisms[x] = ci_test
            else:
                ci_test = self._ci_test_mechanisms[x]

            for nodes in conditioning_set:
                if (
                    ci_test.is_independent(
                        y,
                        "_mechanism",
                        nodes,
                        significance_level=self.significance_level,
                    )
                    is True
                ):
                    return True, set(nodes) | (latent_nodes - {x})

        return False, set()

    def _orient_colliders(
        self,
        temporal_ordering: dict[Hashable, int] = dict(),
    ) -> PDAG:
        """
        Orients the edges that form v-structures in a graph skeleton to form a PDAG.

        For each pair of non-adjacent nodes ``X``, ``Y``, if a common neighbor ``Z``
        is identified as a collider, the edges are oriented as ``X`` -> ``Z`` <- ``Y``.

        When ``orient_rule`` is ``None``, ``Z`` is a collider if it is not in the
        separating set of ``X`` and ``Y``. When ``orient_rule`` is ``"pvalue"`` or
        ``"effect"``, CI tests are run over all subsets of neighbors (MaxP) and
        candidates are sorted by strength before orienting.

        Uses ``self.skeleton_``, ``self.separating_sets_``, ``self.orient_rule``,
        ``self.ci_test_``, ``self.significance_level``, and ``self.max_cond_vars``.

        Parameters
        ----------
        temporal_ordering : dict, optional
            Temporal ordering of variables for filtering collider candidates.

        Returns
        -------
        pgmpy.base.PDAG
            An estimate for the DAG pattern of the BN underlying the data.

        References
        ----------
        - :footcite:t:`neapolitan_2009` (Section 10.1.2, Algorithm 10.2, page 550).
        - :footcite:t:`ramsey_2016`

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.causal_discovery import PC
        >>> from pgmpy.example_models import load_model
        >>> df = load_model("bnlearn/cancer").simulate(int(1e3), seed=42)
        >>> est = PC(ci_test='chi_square').fit(df)
        >>> pdag = est._orient_colliders()
        >>> sorted(pdag.get_edges(data=False))
        [('Pollution', 'Cancer'), ('Xray', 'Cancer')]
        """

        skeleton = self.skeleton_
        separating_sets = self.separating_sets_
        orient_rule = self.orient_rule

        colliders = []
        if orient_rule is None:
            for X, Y in combinations(sorted(skeleton.nodes()), 2):
                if (
                    skeleton.has_edge(X, Y)
                    or X in self.latent_nodes
                    or Y in self.latent_nodes
                ):
                    continue
                sepset = separating_sets.get(frozenset((X, Y)))
                if sepset is None:
                    # This edge was removed by expert knowledge. Ignore.
                    continue
                for Z in sorted(
                    set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y))
                ):
                    if Z not in sepset:
                        colliders.append((X, Y, Z))
        else:
            ci_test = self.ci_test_
            significance_level = self.significance_level
            max_cond_vars = self.max_cond_vars

            candidates = []
            for X, Y in combinations(sorted(skeleton.nodes()), 2):
                if (
                    skeleton.has_edge(X, Y)
                    or X in self.latent_nodes
                    or Y in self.latent_nodes
                ):
                    continue
                if frozenset((X, Y)) not in separating_sets:
                    # This edge was removed by expert knowledge. Ignore.
                    continue
                common_neighbors = set(skeleton.neighbors(X)) & set(
                    skeleton.neighbors(Y)
                )
                if not common_neighbors:
                    continue

                potential = sorted(
                    (set(skeleton.neighbors(X)) - {Y})
                    | (set(skeleton.neighbors(Y)) - {X}),
                    key=repr,
                )

                results = []
                for size in range(min(len(potential), max_cond_vars) + 1):
                    for subset in combinations(potential, size):
                        ci_test(
                            X, Y, list(subset), significance_level=significance_level
                        )
                        results.append((subset, ci_test.p_value_, ci_test.effect_size_))

                for Z in common_neighbors:
                    if orient_rule == "pvalue":
                        max_p_with = max(
                            (p for s, p, _ in results if Z in s), default=-1.0
                        )
                        max_p_without = max(
                            (p for s, p, _ in results if Z not in s), default=-1.0
                        )
                        is_collider = max_p_without > max_p_with
                        priority = max_p_with
                    else:
                        min_eff_with = min(
                            (e for s, _, e in results if Z in s), default=float("inf")
                        )
                        min_eff_without = min(
                            (e for s, _, e in results if Z not in s),
                            default=float("inf"),
                        )
                        is_collider = min_eff_without < min_eff_with
                        priority = -min_eff_with

                    if is_collider:
                        candidates.append((priority, X, Y, Z))

            candidates.sort(key=lambda c: c[0])
            colliders = [(X, Y, Z) for _, X, Y, Z in candidates]

        pdag = PDAG(edge_list=[(u, v, "--") for u, v in skeleton.edges()])
        pdag.add_nodes_from(skeleton.nodes())

        # Orient the edges fixed by the temporal ordering first: an edge between two tiers must
        # point from the earlier tier to the later one. The collider guard below then skips any
        # v-structure that conflicts with these, so temporal order needs no separate check.
        if temporal_ordering:
            for u, v in skeleton.edges():
                if temporal_ordering[u] < temporal_ordering[v]:
                    pdag.orient_undirected_edge(u, v, inplace=True)
                elif temporal_ordering[v] < temporal_ordering[u]:
                    pdag.orient_undirected_edge(v, u, inplace=True)

        for X, Y, Z in colliders:
            # If Z already reaches X or Y through directed edges, orienting X -> Z <- Y would
            # conflict with a committed orientation (including one fixed by the temporal
            # ordering) or close a directed cycle; keep the first orientation and leave these
            # edges for the Meek rules.
            if pdag.has_path(Z, X, edge_types="->") or pdag.has_path(
                Z, Y, edge_types="->"
            ):
                continue

            if pdag.has_edge(X, Z, "--"):
                pdag.orient_undirected_edge(X, Z, inplace=True)
            if pdag.has_edge(Y, Z, "--"):
                pdag.orient_undirected_edge(Y, Z, inplace=True)

        return pdag
