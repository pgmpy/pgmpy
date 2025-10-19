from itertools import permutations
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Optional,
    Set,
    Union,
)

import pandas as pd
from sklearn.utils.validation import validate_data

from pgmpy.base import PDAG, UndirectedGraph
from pgmpy.causal_discovery.base import BaseConstraintCausalDiscovery
from pgmpy.estimators import ExpertKnowledge
from pgmpy.estimators.CITests import get_callable_ci_test


class PC(BaseConstraintCausalDiscovery):
    """
    Constraint-based estimation of DAGs using the PC algorithm.

    Parameters
    ----------
    variant : str, default="parallel"
        PC algorithm variant: {"orig", "stable", "parallel"}.

    ci_test : str or callable, default=None
        Conditional independence test to use.

    return_type : str, default="pdag"
        One of {"dag", "cpdag", "pdag", "skeleton"}.

    significance_level : float, default=0.01
        Threshold for independence tests.

    max_cond_vars : int, default=5
        Max conditioning set size.

    expert_knowledge : ExpertKnowledge or None, default=None
        Prior knowledge about required/forbidden edges.

    enforce_expert_knowledge : bool, default=False
        Whether to enforce expert knowledge during search.

    n_jobs : int, default=-1
        Number of parallel jobs.

    show_progress : bool, default=True
        Whether to show progress bar.

    Attributes
    ----------
    skeleton_ : UndirectedGraph
        An estimate for the undirected graph skeleton of the BN underlying the data.

    separating_sets_ : dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes them
            conditionally independent. (needed for edge orientation procedures)\

    graph_ : PDAG
        The learned causal graph.
    """

    def __init__(
        self,
        variant: str = "parallel",
        ci_test: Optional[Union[str, Callable]] = None,
        return_type: str = "pdag",
        significance_level: float = 0.01,
        max_cond_vars: int = 5,
        expert_knowledge: Optional[ExpertKnowledge] = None,
        enforce_expert_knowledge: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
        self.variant = variant
        self.ci_test = ci_test
        self.return_type = return_type
        self.significance_level = significance_level
        self.max_cond_vars = max_cond_vars
        self.expert_knowledge = expert_knowledge
        self.enforce_expert_knowledge = enforce_expert_knowledge
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def _fit(
        self,
        X: pd.DataFrame,
        y=None,
        independencies=None,
    ):
        """
        Fit data (`X`) and independence relations (optional) to a causal graph. The method
        builds an initial skeleton graph (undirected) based on conditional independence tests.
        Then, the v-structures are oriented based on the separating sets between non-adjacent
        nodes. Finally, Meek's rules are applied to orient as many remaining edges as possible.
        """
        n_samples, n_features = X.shape

        if n_features == 0:
            raise ValueError(
                f"0 feature(s) (shape={X.shape}) while a minimum of 1 is required."
            )
        if n_samples < 2:
            raise ValueError(f"n_samples = {n_samples}, at least 2 are required.")

        # Handle cases like complex data, sparse arrays etc. first
        if isinstance(X, pd.DataFrame):
            _nodes = X.columns
        else:
            _nodes = [f"x{i}" for i in range(X.shape[1])]

        X = validate_data(
            self,
            X=X,
            dtype="numeric",
            accept_sparse=False,
            ensure_all_finite=True,
            reset=True,  # reset=True in fit, reset=False in predict/transform
        )

        X = X.astype(float, copy=False)
        X = pd.DataFrame(X, columns=_nodes)

        # CI test
        ci_test = get_callable_ci_test(self.ci_test, data=X)

        if self.expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()
        else:
            expert_knowledge = self.expert_knowledge

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(X.columns)

        # Step 1: skeleton
        skel, separating_sets = self._build_skeleton(
            data=X,
            independencies=independencies,
            variant=self.variant,
            ci_test=ci_test,
            significance_level=self.significance_level,
            max_cond_vars=self.max_cond_vars,
            expert_knowledge=expert_knowledge,
            enforce_expert_knowledge=self.enforce_expert_knowledge,
            n_jobs=self.n_jobs,
            show_progress=self.show_progress,
        )

        if self.return_type == "skeleton":
            self.graph_ = skel
            return self

        # Step 2: orient colliders
        pdag = self._orient_colliders(
            skel, separating_sets, expert_knowledge.temporal_ordering
        )

        # Step 3: apply rules / expert knowledge
        if expert_knowledge.temporal_order != [[]]:
            pdag = expert_knowledge.apply_expert_knowledge(pdag)
            pdag = pdag.apply_meeks_rules(apply_r4=True)
        elif not self.enforce_expert_knowledge:
            pdag = pdag.apply_meeks_rules(apply_r4=False)
            pdag = expert_knowledge.apply_expert_knowledge(pdag)
            pdag = pdag.apply_meeks_rules(apply_r4=True)
        else:
            pdag = pdag.apply_meeks_rules(apply_r4=False)

        pdag.add_nodes_from(set(X.columns) - set(pdag.nodes()))

        if self.return_type in ("pdag", "cpdag"):
            self.graph_ = pdag
        elif self.return_type == "dag":
            self.graph_ = pdag.to_dag()
        else:
            raise ValueError(
                f"return_type must be one of: dag, pdag, cpdag, skeleton. Got: {self.return_type}"
            )

        return self

    @staticmethod
    def _orient_colliders(
        skeleton: UndirectedGraph,
        separating_sets: Dict[FrozenSet, Set],
        temporal_ordering: Dict[Hashable, int] = dict(),
    ) -> PDAG:
        """
        Orients the edges that form v-structures in a graph skeleton based on
        the `separating_sets` to form a PDAG. For each pair of non adjacent
        nodes `u`, `v` , if a common neighbor `z` is not in the separating set of `u` and `v`;
        then the v-structure is oriented as `u`->`z` , `v`->`z`.

        Parameters
        ----------
        skeleton: nx.Graph
            An undirected graph skeleton as e.g. produced by the
            estimate_skeleton method.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes them
            conditionally independent.

        Returns
        -------
        Model after edge orientation: pgmpy.base.PDAG
            An estimate for the DAG pattern of the BN underlying the data. The
            graph might contain some nodes with both-way edges (X->Y and Y->X).
            Any completion by (removing one of the both-way edges for each such
            pair) results in a I-equivalent Bayesian network DAG.

        References
        ----------
        [1] Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm
                10.2 (page 550)
        [2] http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import PC
        >>> data = pd.DataFrame(
        ...     np.random.randint(0, 4, size=(5000, 3)), columns=list("ABD")
        ... )
        >>> data["C"] = data["A"] - data["B"]
        >>> data["D"] += data["A"]
        >>> c = PC(data)
        >>> pdag = c._orient_colliders(*c._build_skeleton())
        >>> pdag.edges()  # edges: A->C, B->C, A--D (not directed)
        OutEdgeView([('B', 'C'), ('A', 'C'), ('A', 'D'), ('D', 'A')])
        """

        pdag = skeleton.to_directed()

        # 1) for each X-Z-Y, if Z not in the separating set of X,Y, then orient edges
        # as X->Z<-Y (Algorithm 3.4 in Koller & Friedman PGM, page 86)
        for X, Y in permutations(sorted(pdag.nodes()), 2):
            if not skeleton.has_edge(X, Y):
                for Z in set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y)):
                    if Z not in separating_sets[frozenset((X, Y))]:
                        if (temporal_ordering == dict()) or (
                            (temporal_ordering[Z] >= temporal_ordering[X])
                            and (temporal_ordering[Z] >= temporal_ordering[Y])
                        ):
                            pdag.remove_edges_from([(Z, X), (Z, Y)])

        edges = set(pdag.edges())
        undirected_edges = set()
        directed_edges = set()
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.add(tuple(sorted((u, v))))
            else:
                directed_edges.add((u, v))

        pdag_oriented = PDAG(
            directed_ebunch=directed_edges, undirected_ebunch=undirected_edges
        )
        pdag_oriented.add_nodes_from(pdag.nodes())

        return pdag_oriented
