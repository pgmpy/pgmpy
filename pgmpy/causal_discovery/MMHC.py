from collections.abc import Callable

import networkx as nx
import pandas as pd
from sklearn.base import clone

from pgmpy.base import UndirectedGraph
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.causal_discovery.ExpertKnowledge import ExpertKnowledge
from pgmpy.causal_discovery.HillClimbSearch import HillClimbSearch
from pgmpy.ci_tests import BaseCITest, get_ci_test
from pgmpy.structure_score import BaseStructureScore, BDeu
from pgmpy.utils.mathext import powerset


class MMHC(BaseCausalDiscovery):
    """
    Hybrid causal discovery using the Max-Min Parents and Children algorithm
    followed by score-based hill climbing.

    The MMPC phase first estimates an undirected skeleton. Hill climbing then
    orients and selects edges while being restricted to that skeleton.

    Parameters
    ----------
    ci_test : str or callable, default="chi_square"
        Conditional independence test used during MMPC.

    significance_level : float, default=0.01
        Significance level used by the conditional independence tests.

    max_cond_vars : int or None, default=None
        Maximum conditioning-set size used during MMPC. ``None`` preserves the
        original MMPC behavior and tests every subset of the current neighbors.

    scoring_method : str or BaseStructureScore instance, default=None
        Score optimized during the hill-climbing phase. If ``None``, BDeu with
        an equivalent sample size of 10 is used.

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
        Expert knowledge passed to the hill-climbing phase.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph.

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
    >>> data = load_model("bnlearn/asia").simulate(n_samples=1000, seed=42)
    >>> mmhc = MMHC(show_progress=False).fit(data)
    >>> set(mmhc.causal_graph_.nodes()) == set(data.columns)
    True

    References
    ----------
    .. [1] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006).
       The max-min hill-climbing Bayesian network structure learning algorithm.
       Machine Learning, 65, 31-78.
    """

    def __init__(
        self,
        ci_test: str | Callable | None = "chi_square",
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
        """Fit MMHC to ``X`` and store the learned graph attributes."""
        if self.return_type.lower() not in ("dag", "pdag", "cpdag"):
            raise ValueError(f"return_type must be one of: dag, pdag, or cpdag. Got: {self.return_type}")
        if self.max_cond_vars is not None and self.max_cond_vars < 0:
            raise ValueError("max_cond_vars must be non-negative or None.")

        self.ci_test_ = get_ci_test(test=self.ci_test, data=X)
        self.skeleton_ = self._learn_skeleton(
            X,
            ci_test=self.ci_test_,
            significance_level=self.significance_level,
            max_cond_vars=self.max_cond_vars,
        )
        self.causal_graph_ = self._run_hill_climb(X, self.skeleton_)
        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )
        return self

    def _learn_skeleton(
        self,
        X: pd.DataFrame,
        ci_test: BaseCITest | Callable,
        significance_level: float,
        max_cond_vars: int | None = None,
    ) -> UndirectedGraph:
        """Learn the MMPC skeleton."""
        nodes = list(X.columns)

        def association(variable, candidate, conditioning_set):
            if hasattr(ci_test, "run_test"):
                return 1 - ci_test.run_test(variable, candidate, list(conditioning_set))[1]
            return float(
                not ci_test(
                    variable,
                    candidate,
                    list(conditioning_set),
                    significance_level=significance_level,
                )
            )

        def is_independent(variable, candidate, conditioning_set):
            if hasattr(ci_test, "is_independent"):
                return ci_test.is_independent(
                    X=variable,
                    Y=candidate,
                    Z=list(conditioning_set),
                    significance_level=significance_level,
                )
            return ci_test(
                variable,
                candidate,
                list(conditioning_set),
                significance_level=significance_level,
            )

        def minimum_association(variable, candidate, conditioning_set):
            subsets = powerset(conditioning_set)
            if max_cond_vars is not None:
                subsets = (subset for subset in subsets if len(subset) <= max_cond_vars)
            return min(association(variable, candidate, subset) for subset in subsets)

        def max_min_heuristic(variable, neighbors):
            max_min_assoc = 0
            best_candidate = None
            candidates = set(nodes) - set(neighbors + [variable])

            for candidate in candidates:
                min_assoc = minimum_association(variable, candidate, neighbors)
                if min_assoc >= max_min_assoc:
                    best_candidate = candidate
                    max_min_assoc = min_assoc

            return best_candidate, max_min_assoc

        neighbors = {}
        for node in nodes:
            neighbors[node] = []

            while True:
                new_neighbor, new_neighbor_min_assoc = max_min_heuristic(node, neighbors[node])
                if new_neighbor_min_assoc > 0:
                    neighbors[node].append(new_neighbor)
                else:
                    break

            for neighbor in neighbors[node]:
                other_neighbors = [n for n in neighbors[node] if n != neighbor]
                subsets = powerset(other_neighbors)
                if max_cond_vars is not None:
                    subsets = (subset for subset in subsets if len(subset) <= max_cond_vars)
                for separating_set in subsets:
                    if is_independent(node, neighbor, separating_set):
                        neighbors[node].remove(neighbor)
                        break

        for node in nodes:
            for neighbor in neighbors[node]:
                if node not in neighbors[neighbor]:
                    neighbors[node].remove(neighbor)

        skeleton = UndirectedGraph()
        skeleton.add_nodes_from(nodes)
        skeleton.add_edges_from((node, neighbor) for node in nodes for neighbor in neighbors[node])
        return skeleton

    def _run_hill_climb(self, X: pd.DataFrame, skeleton: UndirectedGraph):
        """Run HillClimbSearch restricted to the MMPC skeleton."""
        if self.scoring_method is None:
            scoring_method = BDeu(X, equivalent_sample_size=10)
        else:
            scoring_method = self.scoring_method

        possible_edges = set(nx.complete_graph(n=X.columns, create_using=nx.DiGraph).edges())
        forbidden_edges = possible_edges - set(skeleton.to_directed().edges())

        if self.expert_knowledge is None:
            expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)
        else:
            expert_knowledge = clone(self.expert_knowledge)
            expert_knowledge.forbidden_edges = set(expert_knowledge.forbidden_edges) | forbidden_edges

        hill_climb = HillClimbSearch(
            scoring_method=scoring_method,
            expert_knowledge=expert_knowledge,
            tabu_length=self.tabu_length,
            max_indegree=self.max_indegree,
            return_type="pdag" if self.return_type.lower() == "cpdag" else self.return_type,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            show_progress=self.show_progress,
        )
        return hill_climb.fit(X).causal_graph_
