from collections.abc import Callable, Hashable
from itertools import combinations

import pandas as pd
from sklearn.base import clone

from pgmpy.base import PDAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.causal_discovery._base import BaseCausalDiscovery, _ConstraintMixin
from pgmpy.ci_tests import get_ci_test


class PC(_ConstraintMixin, BaseCausalDiscovery):
    """
    The PC algorithm for causal discovery / structure learning.

    This class implements the PC algorithm [1]_ for causal discovery. Given a
    tabular dataset, the PC algorithm estimates the causal structure among the
    variables in the data in a Directed Acyclic Graph (DAG) or Partially
    Directed Acyclic Graph (PDAG). The algorithm works by identifying
    (conditional) dependencies in data set using statistical independence tests
    and estimates a DAG pattern that satisfies the identified dependencies.

    When expert knowledge is provided, it is always applied as hard directional
    constraints (consistent with :class:`~pgmpy.causal_discovery.HillClimbSearch`):
    forbidden edges never appear in the forbidden direction, required edges are retained
    and oriented as specified, and ``search_space`` (if given) restricts which adjacencies
    may appear. A pair forbidden in both directions (e.g. the complement of a given
    ``search_space``) becomes a non-adjacency, while a single-direction forbidden edge that
    still appears inside a learned collider is left in place with a warning.

    Parameters
    ----------
    variant: str, default="parallel"
        The variant of PC algorithm to run.

        - "orig": The original PC algorithm. Might not give the same results in different runs but does less
                  independence tests compared to stable.
        - "stable": Gives the same result in every run but does needs to do more statistical independence tests.
        - "parallel": Parallel version of PC Stable. Can run on multiple cores with the same result on each run. The
          parallel version would be faster only on datasets with large number of variables or samples. For smaller
          datasets, it might be slower due to the overhead of managing multiple processes.

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

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/alarm")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the PC algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import PC
    >>> pc = PC(variant="parallel", ci_test="chi_square", significance_level=0.01)
    >>> pc.fit(df)
    PC(ci_test='chi_square')
    >>> pc.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.PDAG.PDAG object at 0x...>
    >>> pc.n_features_in_
    37

    Specify expert knowledge:

    References
    ----------
    - :cite:p:`spirtes_glymour_scheines_2001`
    - :cite:p:`neapolitan_2009`
    - :cite:p:`schmidt_2018`
    - :cite:p:`le_2019`
    - :cite:p:`meek_1995`
    - :cite:p:`ramsey_2016`
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

    def _fit(self, X: pd.DataFrame, independencies=None):
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
        self.ci_test_ = get_ci_test(test=self.ci_test, data=X)

        # Check if expert knowledge was specified
        if self.expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()
        else:
            # Clone so the fitted (`*_`) attributes land on a fresh copy, not the user's object.
            expert_knowledge = clone(self.expert_knowledge)

        # Resolve the expert knowledge into its fitted (`*_`) attributes.
        expert_knowledge.fit(X)

        # Step 1: Build the skeleton
        self.skeleton_, self.separating_sets_ = self._build_skeleton(
            data=X,
            independencies=independencies,
            variant=self.variant,
            ci_test=self.ci_test_,
            significance_level=self.significance_level,
            max_cond_vars=self.max_cond_vars,
            expert_knowledge=expert_knowledge,
            n_jobs=self.n_jobs,
            show_progress=self.show_progress,
        )

        # Step 2: Use separating sets to orient colliders
        pdag = self._orient_colliders(
            temporal_ordering=expert_knowledge.temporal_ordering_,
        )

        # Step 3: Apply orientation rules and expert knowledge as hard directional constraints.
        if expert_knowledge.temporal_order is not None:
            pdag = expert_knowledge.apply_to(pdag)
            pdag = pdag.apply_meeks_rules(apply_r4=True)
        else:
            pdag = pdag.apply_meeks_rules(apply_r4=False)
            pdag = expert_knowledge.apply_to(pdag)
            pdag = pdag.apply_meeks_rules(apply_r4=True)

        pdag.add_nodes_from(set(X.columns) - set(pdag.nodes()))

        if self.return_type in ("pdag", "cpdag"):
            self.causal_graph_ = pdag
        elif self.return_type == "dag":
            self.causal_graph_ = pdag.to_dag()
        else:
            raise ValueError(f"return_type must be one of: dag, pdag, or cpdag. Got: {self.return_type}")

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self

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
        - :cite:p:`neapolitan_2009` (Section 10.1.2, Algorithm 10.2, page 550).
        - :cite:p:`ramsey_2016`

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

        pdag = skeleton.to_directed()

        if orient_rule is None:
            for X, Y in combinations(sorted(pdag.nodes()), 2):
                if not skeleton.has_edge(X, Y):
                    sepset = separating_sets.get(frozenset((X, Y)))
                    if sepset is None:
                        # This edge was removed by expert knowledge. Ignore.
                        continue
                    for Z in set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y)):
                        if Z not in sepset:
                            if (temporal_ordering == dict()) or (
                                (temporal_ordering[Z] >= temporal_ordering[X])
                                and (temporal_ordering[Z] >= temporal_ordering[Y])
                            ):
                                if pdag.has_edge(X, Z) and pdag.has_edge(Y, Z):
                                    pdag.remove_edges_from([(Z, X), (Z, Y)])
        else:
            ci_test = self.ci_test_
            significance_level = self.significance_level
            max_cond_vars = self.max_cond_vars

            candidates = []
            for X, Y in combinations(sorted(pdag.nodes()), 2):
                if not skeleton.has_edge(X, Y):
                    if frozenset((X, Y)) not in separating_sets:
                        # This edge was removed by expert knowledge. Ignore.
                        continue
                    common_neighbors = set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y))
                    if not common_neighbors:
                        continue

                    potential = sorted(
                        (set(skeleton.neighbors(X)) - {Y}) | (set(skeleton.neighbors(Y)) - {X}),
                        key=repr,
                    )

                    results = []
                    for size in range(min(len(potential), max_cond_vars) + 1):
                        for subset in combinations(potential, size):
                            ci_test(X, Y, list(subset), significance_level=significance_level)
                            results.append((subset, ci_test.p_value_, ci_test.effect_size_))

                    for Z in common_neighbors:
                        if (temporal_ordering != dict()) and not (
                            temporal_ordering[Z] >= temporal_ordering[X]
                            and temporal_ordering[Z] >= temporal_ordering[Y]
                        ):
                            continue

                        if orient_rule == "pvalue":
                            max_p_with = max((p for s, p, _ in results if Z in s), default=-1.0)
                            max_p_without = max((p for s, p, _ in results if Z not in s), default=-1.0)
                            is_collider = max_p_without > max_p_with
                            priority = max_p_with
                        else:
                            min_eff_with = min((e for s, _, e in results if Z in s), default=float("inf"))
                            min_eff_without = min((e for s, _, e in results if Z not in s), default=float("inf"))
                            is_collider = min_eff_without < min_eff_with
                            priority = -min_eff_with

                        if is_collider:
                            candidates.append((priority, X, Y, Z))

            candidates.sort(key=lambda c: c[0])
            for _, X, Y, Z in candidates:
                if pdag.has_edge(X, Z) and pdag.has_edge(Y, Z):
                    pdag.remove_edges_from([(Z, X), (Z, Y)])

        edges = set(pdag.edges())
        undirected_edges = set()
        directed_edges = set()
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.add(tuple(sorted((u, v))))
            else:
                directed_edges.add((u, v))

        ebunch = [(u, v, "->") for u, v in directed_edges] + [(u, v, "--") for u, v in undirected_edges]
        pdag_oriented = PDAG(edge_list=ebunch)
        pdag_oriented.add_nodes_from(pdag.nodes())

        return pdag_oriented
