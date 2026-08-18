from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix
from sklearn.base import clone

from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.causal_discovery._base import BaseCausalDiscovery


class ILPSearch(BaseCausalDiscovery):
    """
    Exact continuous score-based causal discovery using Integer Linear Programming (ILP).

    This class implements exact continuous score-based causal discovery by solving the Layered
    Network (LN) mixed-integer linear programming formulation via ``scipy.optimize.milp`` (HiGHS solver).

    The algorithm formulates continuous causal graph discovery under linear structural equation models
    (SEMs) as a single global mixed-integer optimization problem. Given observational continuous data
    :math:`X \\in \\mathbb{R}^{n \\times m}`, the formulation solves for edge selection binaries
    :math:`z_{jk} \\in \\{0, 1\\}`, linear structural weights :math:`\\beta_{jk} \\in \\mathbb{R}`, and
    continuous layer potential topological variables :math:`\\psi_j \\in [1, m]`.

    The non-convex acyclicity constraint is guaranteed globally by the linear ordering constraints:

    .. math::

        z_{jk} - (m - 1) z_{kj} - \\psi_k + \\psi_j \\le 0, \\quad \\forall (j, k)

    Sparsity is penalized using either an :math:`L_0` penalty (via auxiliary binary edge activation
    variables :math:`g_{jk}`) or an :math:`L_1` penalty on the structural edge weights.

    The procedure operates in the following steps:
    1. **Superstructure & Search Space Resolution**:
       - If ``expert_knowledge`` is provided, prior required and forbidden edges are resolved, and
         candidate pairs are screened.
       - If ``expert_knowledge`` is ``None``, marginal independence screening (via
         :class:`~pgmpy.causal_discovery.ExpertKnowledge` with ``search_space="marginally_dependent"``)
         is run automatically to identify candidate variable pairs.
    2. **Big-M Bound Estimation**:
       - Computes unconstrained least-squares regression coefficients (:math:`\text{OLS}`) for each variable
         given candidate parents to estimate global Big-M upper bounds :math:`M`.
    3. **MILP Formulation Assembly**:
       - Constructs objective vector ``obj_coefficients``, integrality array, bounds vector, and sparse
         linear constraint matrix ``constraint_matrix`` enforcing tournament relations, Big-M constraints, and
         acyclicity layer ordering.
    4. **Global Exact Solution**:
       - Solves the resulting mixed-integer linear program globally via ``scipy.optimize.milp``.
    5. **Graph Extraction**:
       - Reconstructs optimal directed acyclic graph (:class:`~pgmpy.base.DAG`) with fitted structural
         edge coefficients.

    Parameters
    ----------
    penalty : {"l0", "l1"}, default="l0"
        The regularization penalty type for graph sparsity.

        - ``"l0"``: Penalizes non-zero edges using auxiliary binary activation variables :math:`g_{jk}`
          with objective coefficient ``l_penalty``.
        - ``"l1"``: Penalizes edge presence using an :math:`L_1` penalty weight ``l_penalty``.

    l_penalty : float, default=0.1
        Regularization hyperparameter penalty weight for edge sparsity control. Higher values enforce
        sparser learned graphs.

    return_type : {"dag", "pdag", "cpdag"}, default="dag"
        The graph representation type to return for ``causal_graph_``. If ``"pdag"`` or ``"cpdag"``,
        the learned DAG is converted to a PDAG (essential graph) via ``.to_pdag()``.

    expert_knowledge : ExpertKnowledge or None, default=None
        Prior expert knowledge specifying edge constraints (required edges, forbidden edges, or custom
        candidate search space). If ``None``, marginal independence screening
        (:class:`~pgmpy.causal_discovery.ExpertKnowledge` with ``search_space="marginally_dependent"``)
        is automatically used to screen candidate pairs.

    options : dict or None, default=None
        Dictionary of solver options passed directly to ``scipy.optimize.milp``.
        Recognized options for the underlying solver include:

        - ``disp`` : bool, default=False
            If ``True``, prints optimization status indicators to the console during search.
        - ``node_limit`` : int, optional
            The maximum number of nodes (linear program relaxations) to solve before stopping.
        - ``presolve`` : bool, default=True
            Presolve attempts to identify trivial infeasibilities, unboundedness, and simplify problem.
        - ``time_limit`` : float, optional
            The maximum number of seconds allotted to solve the problem (e.g., ``{"time_limit": 60.0}``).
        - ``mip_rel_gap`` : float, optional
            Termination criterion for MIP solver: terminates when gap between primal objective value
            and dual objective bound is :math:`\\le \\text{mip\\_rel\\_gap}` (default: 0.0001).

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG or pgmpy.base.PDAG
        The learned optimal causal graph as a DAG or PDAG instance.

    adjacency_matrix_ : pd.DataFrame
        Binary adjacency matrix representation of ``causal_graph_``.

    n_features_in_ : int
        The number of features (variables) in the dataset used for fitting.

    feature_names_in_ : np.ndarray
        The feature names in the dataset used for fitting.

    milp_result_ : scipy.optimize.OptimizeResult
        The raw optimization result returned by the underlying ``scipy.optimize.milp`` solver.
        Contains solver status, objective value, and full decision variable array.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import ExpertKnowledge, ILPSearch
    >>> df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 4.0, 6.0]})
    >>> ek = ExpertKnowledge(required_edges=[("A", "B")])
    >>> ilp = ILPSearch(expert_knowledge=ek, options={"time_limit": 10.0})
    >>> ilp = ilp.fit(df)
    >>> ("A", "B") in ilp.causal_graph_.edges()
    True

    References
    ----------
    - :footcite:t:`manzour_2021`
    """

    def __init__(
        self,
        penalty: str = "l0",
        l_penalty: float = 0.1,
        return_type: str = "dag",
        expert_knowledge: ExpertKnowledge | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.penalty = penalty
        self.l_penalty = l_penalty
        self.return_type = return_type
        self.expert_knowledge = expert_knowledge
        self.options = options

    def _fit(self, X: pd.DataFrame) -> ILPSearch:
        """
        Estimate the causal DAG structure from tabular dataset X using MILP optimization.

        Parameters
        ----------
        X : pd.DataFrame
            Observational dataset with continuous columns.

        Returns
        -------
        self : ILPSearch
            Fitted estimator with attributes ``causal_graph_`` and ``adjacency_matrix_`` set.
        """
        # Step 0: Continuous Data Validation
        try:
            X.astype(float)
        except (ValueError, TypeError):
            raise ValueError("ILPSearch requires continuous (numeric) variables.")

        variable_map = {name: idx for idx, name in enumerate(self.feature_names_in_)}

        # Step 1: Superstructure & Expert Knowledge Resolution via CI-test screening
        if self.expert_knowledge is not None:
            ek = cast(ExpertKnowledge, clone(self.expert_knowledge)).fit(X)
            required_edges = set(ek.required_edges_)
            forbidden_edges = set(ek.forbidden_edges_)
        else:
            required_edges = set()
            forbidden_edges = set()

        if self.expert_knowledge is not None and self.expert_knowledge.search_space:
            search_space = set(ek.search_space_)
        else:
            search_space = set(ExpertKnowledge(search_space="marginally_dependent").fit(X).search_space_)

        candidate_pairs = (search_space | required_edges) - forbidden_edges

        # Build candidate directed edge list
        directed_edges = [(variable_map[u], variable_map[v]) for u, v in candidate_pairs if u != v]
        num_directed_edges = len(directed_edges)
        directed_edge_index = {edge: i for i, edge in enumerate(directed_edges)}

        # Step 2: Big-M Estimation via unconstrained OLS
        X_mat_raw = X.to_numpy(dtype=float)
        X_mat = X_mat_raw - X_mat_raw.mean(axis=0)

        max_beta = 0.0
        for k in range(self.n_features_in_):
            parents = [j for (j, target) in directed_edges if target == k]
            if parents:
                X_p = X_mat[:, parents]
                y = X_mat[:, k]
                beta_ols, _, _, _ = np.linalg.lstsq(X_p, y, rcond=None)
                max_beta = max(max_beta, float(np.max(np.abs(beta_ols))))
        M = max(2.0 * max_beta, 10.0)

        # Step 3: Decision Vector Assembly
        # Layout: x = [z (num_edges), beta (num_edges), g (num_edges if L0), psi (n_features_in_)]
        has_g = self.penalty == "l0"
        offset_z = 0
        offset_beta = offset_z + num_directed_edges
        offset_g = offset_beta + num_directed_edges if has_g else offset_beta
        offset_psi = (offset_g + num_directed_edges) if has_g else (offset_beta + num_directed_edges)
        n_solver_vars = offset_psi + self.n_features_in_

        # Step 4: Objective Vector obj_coefficients
        obj_coefficients = np.zeros(n_solver_vars)
        n_samples = len(X_mat)
        for idx, (j, k) in enumerate(directed_edges):
            y = X_mat[:, k]
            rss0 = float(np.sum(y**2))
            X_j = X_mat[:, [j]]
            beta_ols, _, _, _ = np.linalg.lstsq(X_j, y, rcond=None)
            rss1 = float(np.sum((y - X_j @ beta_ols) ** 2))
            delta_s = (rss0 - rss1) / float(n_samples) if n_samples > 0 else 0.0

            act_idx = (offset_g + idx) if has_g else (offset_z + idx)
            obj_coefficients[act_idx] = -delta_s + self.l_penalty

        # Integrality: 1 for z and g (binary), 0 for beta and psi (continuous)
        integrality = np.zeros(n_solver_vars)
        integrality[offset_z : offset_z + num_directed_edges] = 1
        if has_g:
            integrality[offset_g : offset_g + num_directed_edges] = 1

        # Bounds
        lb = np.zeros(n_solver_vars)
        ub = np.zeros(n_solver_vars)
        # z in [0, 1]
        lb[offset_z : offset_z + num_directed_edges] = 0
        ub[offset_z : offset_z + num_directed_edges] = 1
        # beta in [-M, M]
        lb[offset_beta : offset_beta + num_directed_edges] = -M
        ub[offset_beta : offset_beta + num_directed_edges] = M
        # g in [0, 1] if L0
        if has_g:
            lb[offset_g : offset_g + num_directed_edges] = 0
            ub[offset_g : offset_g + num_directed_edges] = 1
        # psi in [1, n_features_in_]
        lb[offset_psi : offset_psi + self.n_features_in_] = 1
        ub[offset_psi : offset_psi + self.n_features_in_] = self.n_features_in_

        # Enforce expert knowledge constraints by fixing decision variable bounds:
        # - Required edges: lower bound set to 1 (forces edge presence)
        # - Forbidden edges: upper bound set to 0 (prohibits edge creation)
        for u, v in required_edges:
            if (variable_map[u], variable_map[v]) in directed_edge_index:
                idx = directed_edge_index[(variable_map[u], variable_map[v])]
                lb[offset_z + idx] = 1
                if has_g:
                    lb[offset_g + idx] = 1
        for u, v in forbidden_edges:
            if (variable_map[u], variable_map[v]) in directed_edge_index:
                idx = directed_edge_index[(variable_map[u], variable_map[v])]
                ub[offset_z + idx] = 0
                if has_g:
                    ub[offset_g + idx] = 0

        var_bounds = Bounds(cast(Any, lb), cast(Any, ub))

        # Step 5: Build Constraints Sparse Matrix constraint_matrix
        constraint_rows = []
        constraint_lb = []
        constraint_ub = []

        # (A) Tournament: z_jk + z_kj = 1 (for pairs in directed_edges)
        seen_pairs = set()
        for j, k in directed_edges:
            pair = tuple(sorted((j, k)))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                row = np.zeros(n_solver_vars)
                row[offset_z + directed_edge_index[(j, k)]] = 1
                if (k, j) in directed_edge_index:
                    row[offset_z + directed_edge_index[(k, j)]] = 1
                constraint_rows.append(row)
                constraint_lb.append(1.0)
                constraint_ub.append(1.0)

        # (B) Big-M bounds: -M * active <= beta <= M * active
        # active = g_jk if L0 else z_jk
        for j, k in directed_edges:
            idx = directed_edge_index[(j, k)]
            act_idx = (offset_g + idx) if has_g else (offset_z + idx)
            beta_idx = offset_beta + idx

            # beta - M * active <= 0
            row1 = np.zeros(n_solver_vars)
            row1[beta_idx] = 1
            row1[act_idx] = -M
            constraint_rows.append(row1)
            constraint_lb.append(-np.inf)
            constraint_ub.append(0.0)

            # -beta - M * active <= 0
            row2 = np.zeros(n_solver_vars)
            row2[beta_idx] = -1
            row2[act_idx] = -M
            constraint_rows.append(row2)
            constraint_lb.append(-np.inf)
            constraint_ub.append(0.0)

            # Link g_jk <= z_jk if L0
            if has_g:
                row_link = np.zeros(n_solver_vars)
                row_link[offset_g + idx] = 1
                row_link[offset_z + idx] = -1
                constraint_rows.append(row_link)
                constraint_lb.append(-np.inf)
                constraint_ub.append(0.0)

        # (C) Layer Acyclicity: z_jk - (m-1)*z_kj - psi_k + psi_j <= 0
        for j, k in directed_edges:
            row = np.zeros(n_solver_vars)
            row[offset_z + directed_edge_index[(j, k)]] = 1
            if (k, j) in directed_edge_index:
                row[offset_z + directed_edge_index[(k, j)]] = -(self.n_features_in_ - 1)
            row[offset_psi + j] = 1
            row[offset_psi + k] = -1
            constraint_rows.append(row)
            constraint_lb.append(-np.inf)
            constraint_ub.append(0.0)

        constraint_matrix = csc_matrix(constraint_rows) if constraint_rows else csc_matrix((0, n_solver_vars))
        linear_constraints = LinearConstraint(constraint_matrix, cast(Any, constraint_lb), cast(Any, constraint_ub))

        # Step 6: Call SciPy MILP Solver
        res = milp(
            c=obj_coefficients,
            integrality=integrality,
            bounds=var_bounds,
            constraints=linear_constraints,
            options=self.options,
        )

        if not res.success and res.x is None:
            raise RuntimeError(f"ILP optimization failed: {res.status} ({res.message})")

        self.milp_result_ = res

        # Step 7: Extract Graph
        sol_z = res.x[offset_z : offset_z + num_directed_edges]
        sol_act = res.x[offset_g : offset_g + num_directed_edges] if has_g else sol_z

        dag = DAG()
        dag.add_nodes_from(self.feature_names_in_)

        for idx, (j, k) in enumerate(directed_edges):
            if sol_act[idx] > 0.5:
                dag.add_edge(self.feature_names_in_[j], self.feature_names_in_[k])

        if self.return_type in ("pdag", "cpdag"):
            self.causal_graph_ = dag.to_pdag()
        else:
            self.causal_graph_ = dag

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.feature_names_in_)
        )
        return self
