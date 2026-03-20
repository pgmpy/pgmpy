from collections import deque
from collections.abc import Hashable

import networkx as nx
import pandas as pd
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery, _ScoreMixin
from pgmpy.estimators import ExpertKnowledge
from pgmpy.estimators.StructureScore import StructureScore, get_scoring_method


class HillClimbSearch(_ScoreMixin, _BaseCausalDiscovery):
    """
    Score-based causal discovery using hill climbing optimization.

    This class implements the HillClimbSearch algorithm [1]_ for causal discovery.
    Given a tabular dataset, the algorithm estimates the causal structure among
    the variables in the data as a Directed Acyclic Graph (DAG). The algorithm
    works by iteratively making local modifications to the graph structure
    (adding, removing, or reversing edges) and keeping changes that improve
    the score until a local maximum is reached.

    The algorithm is a greedy local search method that:
    1. Starts from an initial graph (empty by default or based on provided expert knowledge).
    2. Evaluates all possible single-edge modifications (add, delete, reverse).
    3. Applies the modification with the highest score improvement.
    4. Repeats until no improvement can be made.

    A tabu list is used to prevent the algorithm from immediately undoing recent
    changes, which helps avoid getting stuck in local optima.

    Parameters
    ----------
    scoring_method : str or StructureScore instance, default=None
        The score to be optimized during structure estimation. Supported
        structure scores:

        - Discrete data: 'k2', 'bdeu', 'bds', 'bic-d', 'aic-d'
        - Continuous data: 'll-g', 'aic-g', 'bic-g'
        - Mixed data: 'll-cg', 'aic-cg', 'bic-cg'

        If None, the appropriate scoring method is automatically selected based
        on the data type. Also accepts a custom score instance that inherits
        from `StructureScore`.

    start_dag : DAG instance, default=None
        The starting point for the local search. By default, a completely
        disconnected network (no edges) is used. If provided, the DAG must
        contain exactly the same variables as in the data.

    tabu_length : int, default=100
        The number of recent graph modifications to store in the tabu list.
        These modifications cannot be reversed during the search procedure.
        This serves to enforce a wider exploration of the search space.

    max_indegree : int or None, default=None
        If provided, the procedure only searches among models where all nodes
        have at most `max_indegree` parents. This can significantly reduce
        the search space and computation time for large graphs.

    expert_knowledge : ExpertKnowledge instance, default=None
        Expert knowledge to be used with the algorithm. Expert knowledge
        allows specification of:

        - Required edges that must be present in the final graph
        - Forbidden edges that cannot be present in the final graph
        - Temporal ordering of nodes

    return_type : str, default='pdag'
        The type of graph to return. Options are:
        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG) where edges that
          could not be oriented are left undirected.

    epsilon : float, default=1e-4
        Defines the exit condition. If the improvement in score is less
        than `epsilon`, the algorithm terminates and returns the learned model.

    max_iter : int, default=1e6
        The maximum number of iterations allowed. The algorithm terminates
        and returns the learned model when the number of iterations exceeds
        `max_iter`.

    use_cache : bool, default=True
        If True, uses caching of local scores for faster computation.
        Note: Caching only works for scoring methods which are decomposable.
        Can give incorrect results for custom non-decomposable scoring methods.

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned causal graph as a DAG at a (local) score maximum.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.utils import get_example_model
    >>> model = get_example_model("alarm")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the HillClimbSearch algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import HillClimbSearch
    >>> hc = HillClimbSearch(scoring_method="bic-d")
    >>> hc.fit(df)
    >>> hc.causal_graph_.edges()

    Use expert knowledge to constrain the search:

    >>> from pgmpy.estimators import ExpertKnowledge
    >>> expert = ExpertKnowledge(forbidden_edges=[("HISTORY", "CVP")])
    >>> hc = HillClimbSearch(scoring_method="bic-d", expert_knowledge=expert)
    >>> hc.fit(df)

    References
    ----------
    .. [1] Koller & Friedman, Probabilistic Graphical Models - Principles and
           Techniques, 2009, Section 18.4.3 (page 811ff)
    """

    def __init__(
        self,
        scoring_method: str | StructureScore | None = None,
        start_dag: DAG | None = None,
        tabu_length: int = 100,
        max_indegree: int | None = None,
        expert_knowledge: ExpertKnowledge | None = None,
        return_type: str = "pdag",
        epsilon: float = 1e-4,
        max_iter: int = int(1e6),
        use_cache: bool = True,
        show_progress: bool = True,
    ):
        self.scoring_method = scoring_method
        self.start_dag = start_dag
        self.tabu_length = tabu_length
        self.max_indegree = max_indegree
        self.expert_knowledge = expert_knowledge
        self.return_type = return_type
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.use_cache = use_cache
        self.show_progress = show_progress

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the HillClimbSearch algorithm.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The data to learn the causal structure from. If a numpy array is
            passed, then the column names would be integers from 0 to n_features-1.

        Returns
        -------
        self : pgmpy.causal_discovery.HillClimbSearch
            Returns the instance with the fitted attributes.
        """
        self.variables_ = list(X.columns)

        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check scoring_method
        score, score_c = get_scoring_method(self.scoring_method, X, self.use_cache)
        score_fn = score_c.local_score

        # Step 1.2: Check the start_dag
        if self.start_dag is None:
            start_dag = DAG()
            start_dag.add_nodes_from(self.variables_)
        elif not isinstance(self.start_dag, DAG) or not set(self.start_dag.nodes()) == set(self.variables_):
            raise ValueError("'start_dag' should be a DAG with the same variables as the data set, or 'None'.")
        else:
            start_dag = self.start_dag.copy()

        # Step 1.3: Check if expert knowledge was specified
        if self.expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()
        else:
            expert_knowledge = self.expert_knowledge

        # Step 1.3.1: If search_space in expert_knowledge is not None, limit the search space
        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(X.columns)

        # Step 1.4: Check if required edges cause a cycle
        start_dag.add_edges_from(expert_knowledge.required_edges)
        if not nx.is_directed_acyclic_graph(start_dag):
            raise ValueError(
                "required_edges create a cycle in start_dag. Please modify either required_edges or start_dag."
            )
        expert_knowledge._orient_temporal_forbidden_edges(start_dag, only_edges=False)
        start_dag.remove_edges_from(expert_knowledge.forbidden_edges)

        # Step 1.5: Initialize max_indegree, tabu_list, and progress bar
        max_indegree = self.max_indegree
        if max_indegree is None:
            max_indegree = float("inf")

        tabu_list: deque[tuple[str, tuple[Hashable, Hashable]]] = deque(maxlen=self.tabu_length)
        current_model = start_dag

        if self.show_progress and config.SHOW_PROGRESS:
            iteration = trange(int(self.max_iter))
        else:
            iteration = range(int(self.max_iter))

        # Step 2: For each iteration, find the best scoring operation and
        #         do that to the current model. If no legal operation is
        #         possible, sets best_operation=None.
        for _ in iteration:
            best_operation, best_score_delta = max(
                self._legal_operations_dag(
                    model=current_model,
                    score=score_fn,
                    structure_score=score.structure_prior_ratio,
                    tabu_list=tabu_list,
                    max_indegree=max_indegree,
                    forbidden_edges=expert_knowledge.forbidden_edges,
                    required_edges=expert_knowledge.required_edges,
                ),
                key=lambda t: t[1],
                default=(None, None),
            )

            if best_operation is None or best_score_delta < self.epsilon:
                break
            elif best_operation[0] == "+":
                current_model.add_edge(*best_operation[1])
                tabu_list.append(("-", best_operation[1]))
            elif best_operation[0] == "-":
                current_model.remove_edge(*best_operation[1])
                tabu_list.append(("+", best_operation[1]))
            elif best_operation[0] == "flip":
                X_node, Y_node = best_operation[1]
                current_model.remove_edge(X_node, Y_node)
                current_model.add_edge(Y_node, X_node)
                tabu_list.append(best_operation)

        # Step 3: Store results
        if self.return_type.lower() == "dag":
            self.causal_graph_ = current_model
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = current_model.to_pdag()
        else:
            raise ValueError(f"return_type must be one of: dag, pdag, or cpdag. Got: {self.return_type}")

        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, weight=1, dtype="int")

        return self
