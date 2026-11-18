from collections.abc import Callable
from itertools import permutations

import pandas as pd
from sklearn.base import clone

from pgmpy.base import UndirectedGraph
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.causal_discovery.ExpertKnowledge import ExpertKnowledge
from pgmpy.causal_discovery.HillClimbSearch import HillClimbSearch
from pgmpy.ci_tests import BaseCITest, get_ci_test
from pgmpy.structure_score import BaseStructureScore, get_scoring_method
from pgmpy.utils.mathext import powerset


class MMHC(BaseCausalDiscovery):
    """
    Hybrid causal discovery using the Max-Min Parents and Children algorithm
    followed by score-based hill climbing.

    The MMPC phase first estimates an undirected skeleton. Hill climbing then
    orients and selects edges while being restricted to that skeleton.

    Parameters
    ----------
    ci_test : str, BaseCITest instance, callable, or None, default=None
        Conditional independence test used during MMPC. If ``None``, the default
        test for the data type is used (e.g. chi-square for discrete data). A
        candidate's association with a node is 0 when the test finds them
        independent, and the test's ``effect_size_`` otherwise. Unlike p-values,
        effect sizes do not underflow to a tie for strong dependencies. Plain
        callables return no effect size, so every candidate they find dependent
        gets an association of 1.

    significance_level : float, default=0.01
        Significance level used by the conditional independence tests.

    max_cond_vars : int or None, default=None
        Maximum conditioning-set size used during MMPC. ``None`` preserves the
        original MMPC behavior and tests every subset of the current neighbors.

    scoring_method : str or BaseStructureScore instance, default=None
        Score optimized during the hill-climbing phase. If ``None``, the default
        score for the data type is used (e.g. BIC for discrete data).

    tabu_length : int, default=10
        Number of recent graph modifications kept in the tabu list.

    max_indegree : int or None, default=None
        Maximum number of parents allowed for any node during hill climbing.

    max_iter : int, default=1e6
        Maximum number of hill-climbing iterations.

    epsilon : float, default=1e-4
        Minimum score improvement required for another hill-climbing iteration.

    return_type : str, default="pdag"
        Type of graph stored in ``causal_graph_``. Must be ``"dag"``,
        ``"pdag"``, or ``"cpdag"``. ``"cpdag"`` is treated as ``"pdag"``.

    show_progress : bool, default=True
        Whether to show progress during hill climbing.

    expert_knowledge : ExpertKnowledge instance, default=None
        Expert knowledge used in both phases. In MMPC, required edges are always
        in the skeleton, pairs forbidden in both directions never are, and a
        temporal order limits the conditioning sets. Hill climbing then applies
        all the forbidden and required edges.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph.

    ci_test_ : BaseCITest or callable
        The conditional independence test used during MMPC.

    scoring_method_ : BaseStructureScore
        The score used during hill climbing.

    adjacency_matrix_ : pandas.DataFrame
        Adjacency matrix of the learned causal graph.

    skeleton_ : pgmpy.base.UndirectedGraph
        Skeleton learned during the MMPC phase.

    n_features_in_ : int
        Number of features in the data used to fit the estimator.

    feature_names_in_ : pandas.Index
        Names of the features in the data used to fit the estimator.

    Examples
    --------
    >>> from pgmpy.causal_discovery import MMHC
    >>> from pgmpy.example_models import load_model
    >>> data = load_model("bnlearn/asia").simulate(n_samples=1000, rng=42)
    >>> mmhc = MMHC(show_progress=False).fit(data)
    >>> set(mmhc.causal_graph_.nodes()) == set(data.columns)
    True

    References
    ----------
    - :footcite:t:`tsamardinos_2006`
    """

    _tags = {
        "name": "mmhc",
        "data_types": ["discrete", "continuous", "mixed"],
        "identifiable_graph": "cpdag",
        "requires_target": False,
        "capability:multivariate": True,
        "capability:expert_knowledge": ["forbidden_edges", "required_edges", "search_space", "temporal_order"],
        "assumption:causal_sufficiency": True,
        "assumption:acyclicity": True,
        "assumption:faithfulness": True,
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
        "assumption:non_gaussian_noise": False,
        "assumption:low_noise": False,
    }

    def __init__(
        self,
        ci_test: str | Callable | None = None,
        significance_level: float = 0.01,
        max_cond_vars: int | None = None,
        scoring_method: str | BaseStructureScore | None = None,
        tabu_length: int = 10,
        max_indegree: int | None = None,
        max_iter: int = int(1e6),
        epsilon: float = 1e-4,
        return_type: str = "pdag",
        show_progress: bool = True,
        expert_knowledge: ExpertKnowledge | None = None,
    ) -> None:
        self.ci_test = ci_test
        self.significance_level = significance_level
        self.max_cond_vars = max_cond_vars
        self.scoring_method = scoring_method
        self.tabu_length = tabu_length
        self.max_indegree = max_indegree
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.return_type = return_type
        self.show_progress = show_progress
        self.expert_knowledge = expert_knowledge

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the MMHC algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.MMHC
            Returns the instance with the fitted attributes.
        """
        # Step 0: Validate the arguments.
        if self.return_type.lower() not in ("dag", "pdag", "cpdag"):
            raise ValueError(f"return_type must be one of: dag, pdag, or cpdag. Got: {self.return_type}")
        if self.max_cond_vars is not None and self.max_cond_vars < 0:
            raise ValueError("max_cond_vars must be non-negative or None.")

        # Step 1: Resolve the CI test, the score and the expert knowledge.
        # Step 1.1: Resolve the CI test and the score.
        self.ci_test_ = get_ci_test(test=self.ci_test, data=X)
        self.scoring_method_ = get_scoring_method(self.scoring_method, X)

        # Step 1.2: Fit a copy of the expert knowledge, leaving the user's object unchanged.
        if self.expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()
        else:
            expert_knowledge = clone(self.expert_knowledge)
        expert_knowledge.fit(X)

        # Step 2: Learn the skeleton using MMPC.
        self.skeleton_ = self._learn_skeleton(
            data=X,
            ci_test=self.ci_test_,
            significance_level=self.significance_level,
            max_cond_vars=self.max_cond_vars,
            expert_knowledge=expert_knowledge,
        )

        # Step 3: Restrict hill climbing to the skeleton by forbidding every edge outside it.
        outside_skeleton = set(permutations(X.columns, 2)) - set(self.skeleton_.to_directed().edges())
        hill_climb_knowledge = ExpertKnowledge(
            forbidden_edges=expert_knowledge.forbidden_edges_ | outside_skeleton,
            required_edges=expert_knowledge.required_edges_,
        )

        # Step 4: Orient and select the skeleton edges using hill climbing.
        hill_climb = HillClimbSearch(
            scoring_method=self.scoring_method_,
            expert_knowledge=hill_climb_knowledge,
            tabu_length=self.tabu_length,
            max_indegree=self.max_indegree,
            return_type="pdag" if self.return_type.lower() == "cpdag" else self.return_type,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            show_progress=self.show_progress,
        ).fit(X)

        # Step 5: Store the results.
        self.causal_graph_ = hill_climb.causal_graph_
        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self

    def _learn_skeleton(
        self,
        data: pd.DataFrame,
        ci_test: BaseCITest | Callable,
        significance_level: float,
        max_cond_vars: int | None = None,
        expert_knowledge: ExpertKnowledge | None = None,
    ) -> UndirectedGraph:
        """
        Estimates the skeleton using the Max-Min Parents and Children (MMPC) algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            The data to learn the skeleton from.

        ci_test : BaseCITest instance or callable
            The conditional independence test to use.

        significance_level : float
            The significance level used by the conditional independence tests.

        max_cond_vars : int or None, default=None
            The maximum size of the conditioning sets. ``None`` tests every subset of the current neighbors.

        expert_knowledge : ExpertKnowledge instance or None, default=None
            Fitted expert knowledge. Required edges are always in the skeleton, and pairs forbidden in both directions
            never are. With a temporal order, conditioning sets only contain variables from tiers no later than the
            later of the two tested variables.

        Returns
        -------
        skeleton : pgmpy.base.UndirectedGraph
            The estimated skeleton.

        References
        ----------
        - :footcite:t:`tsamardinos_2006` (Algorithms 1 and 2)
        """
        # Step 1: Read the tiers, the required pairs and the pairs forbidden in both directions from the expert
        #         knowledge, and start each node's neighbors with its required neighbors.
        nodes = list(data.columns)
        if expert_knowledge is None:
            temporal_ordering, required_edges, forbidden_edges = {}, set(), set()
        else:
            temporal_ordering = expert_knowledge.temporal_ordering_
            required_edges = expert_knowledge.required_edges_
            forbidden_edges = expert_knowledge.forbidden_edges_

        tier = temporal_ordering or dict.fromkeys(nodes, 0)
        required_pairs = {frozenset(edge) for edge in required_edges}
        forbidden_pairs = {frozenset((u, v)) for u, v in forbidden_edges if (v, u) in forbidden_edges}
        neighbors = {node: [other for other in nodes if frozenset((node, other)) in required_pairs] for node in nodes}

        # Step 2: Find the candidate parents and children (the neighbors) of each node.
        for node in nodes:
            # Step 2.1: Forward phase. Add the candidate whose minimum association with `node`, over all subsets of
            #           the current neighbors, is the largest. Stop when every remaining candidate is independent of
            #           `node` given some subset, i.e. its minimum association is 0.
            candidates = [
                candidate
                for candidate in nodes
                if candidate != node
                and candidate not in neighbors[node]
                and frozenset((node, candidate)) not in forbidden_pairs
            ]
            min_assoc = dict.fromkeys(candidates, float("inf"))
            added = None
            while True:
                # Step 2.1.1: Update each candidate's minimum association with the conditioning sets that contain the
                #             last added neighbor.
                for candidate in candidates:
                    conditioning_variables = [n for n in neighbors[node] if tier[n] <= max(tier[node], tier[candidate])]
                    for conditioning_set in powerset(conditioning_variables, max_size=max_cond_vars):
                        if added is not None and added not in conditioning_set:
                            continue
                        if ci_test(node, candidate, list(conditioning_set), significance_level=significance_level):
                            min_assoc[candidate] = 0.0
                            break
                        association = ci_test.effect_size_ if isinstance(ci_test, BaseCITest) else 1.0
                        min_assoc[candidate] = min(min_assoc[candidate], association)

                # Step 2.1.2: Drop the candidates with a minimum association of 0, and add the remaining one with the
                #             largest minimum association as a neighbor.
                candidates = [candidate for candidate in candidates if min_assoc[candidate] > 0]
                if not candidates:
                    break

                added = max(candidates, key=min_assoc.__getitem__)
                neighbors[node].append(added)
                candidates.remove(added)

            # Step 2.2: Backward phase. Remove the neighbors, other than required ones, that are independent of `node`
            #           given some subset of the other neighbors found in the forward phase.
            forward_neighbors = list(neighbors[node])
            for neighbor in forward_neighbors:
                if frozenset((node, neighbor)) in required_pairs:
                    continue

                other_neighbors = [
                    n for n in forward_neighbors if n != neighbor and tier[n] <= max(tier[node], tier[neighbor])
                ]
                for separating_set in powerset(other_neighbors, max_size=max_cond_vars):
                    if ci_test(node, neighbor, list(separating_set), significance_level=significance_level):
                        neighbors[node].remove(neighbor)
                        break

        # Step 3: Symmetry correction. Keep `neighbor` as a neighbor of `node` only if `node` is also a neighbor of
        #         `neighbor`.
        for node in nodes:
            for neighbor in list(neighbors[node]):
                if node not in neighbors[neighbor]:
                    neighbors[node].remove(neighbor)

        # Step 4: Build the skeleton from the neighbors.
        skeleton = UndirectedGraph()
        skeleton.add_nodes_from(nodes)
        skeleton.add_edges_from((node, neighbor) for node in nodes for neighbor in neighbors[node])

        return skeleton
