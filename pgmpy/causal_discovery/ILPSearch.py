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
from pgmpy.utils import get_dataset_type


class ILPSearch(BaseCausalDiscovery):
    """
    Exact continuous score-based causal discovery using Integer Linear Programming (ILP).

    This class implements exact continuous score-based causal discovery by solving the Layered
    Network (LN) mixed-integer linear programming formulation via ``scipy.optimize.milp`` (HiGHS solver).

    The original formulation in :footcite:t:`manzour_2021` is a Mixed-Integer Quadratic Program (MIQP),
    where the objective minimizes the residual sum of squares (RSS) jointly over continuous structural
    weights :math:`\\beta_{jk}` and binary edge indicators :math:`z_{jk}`:

    .. math::

        \\min_{z, \\beta} \\frac{1}{n} \\|X - X B\\|_F^2 + \\lambda \\sum_{j,k} z_{jk}

    Since ``scipy.optimize.milp`` only supports **linear** objective functions, we cannot directly
    optimize a quadratic RSS objective inside the solver. Instead, we **pre-compute** the marginal RSS
    improvement :math:`\\Delta S_{jk}` for each candidate edge :math:`(j, k)` using unconstrained
    ordinary least squares (OLS) regression *before* the solver runs:

    .. math::

        \\Delta S_{jk} = \\frac{1}{n} \\left( \\text{RSS}_{\\emptyset}^{(k)} - \\text{RSS}_{\\{j\\}}^{(k)} \\right)

    This converts the quadratic objective into a purely linear one over binary activation variables:

    .. math::

        \\min_{z} \\sum_{j,k} \\left( -\\Delta S_{jk} + \\lambda \\right) z_{jk}

    This pre-computation makes the problem solvable by a standard MILP solver, but it introduces
    a marginal (single-parent) approximation — each edge is scored independently rather than jointly
    with other parents.

    .. note::

        **Why L1 regularization is not supported:**

        The paper also proposes an :math:`L_1` (Lasso) penalty that penalizes the continuous structural
        edge weights :math:`\\sum_{j,k} |\\beta_{jk}|`. Implementing this inside a linear ILP solver
        would require jointly optimizing continuous weight variables :math:`\\beta_{jk}` alongside
        the discrete DAG structure. Since the RSS objective :math:`\\|X - XB\\|_F^2` is quadratic
        in :math:`\\beta`, this turns the problem into a Mixed-Integer Quadratic Program (MIQP),
        which ``scipy.optimize.milp`` cannot solve. An iterative approach (solving the LP, extracting
        weights, re-linearizing, and re-solving) would compromise optimality guarantees and
        significantly increase computation time, defeating the purpose of exact optimization.
        Therefore, only the :math:`L_0` penalty is implemented.

    The algorithm formulates continuous causal graph discovery under linear structural equation models
    (SEMs) as a single global mixed-integer optimization problem. Given observational continuous data
    :math:`X \\in \\mathbb{R}^{n \\times m}`, the formulation solves for edge orientation binaries
    :math:`z_{jk} \\in \\{0, 1\\}`, edge selection binaries :math:`g_{jk} \\in \\{0, 1\\}`, structural
    weights :math:`\\beta_{jk} \\in \\mathbb{R}`, and continuous layer potential topological variables
    :math:`\\psi_j \\in [1, m]`.

    The non-convex acyclicity constraint is guaranteed globally by the linear ordering constraints:

    .. math::

        z_{jk} - (m - 1) z_{kj} - \\psi_k + \\psi_j \\le 0, \\quad \\forall (j, k)

    The procedure operates in the following steps:
    1. **Superstructure & Search Space Resolution**:
       - If ``expert_knowledge`` is provided, prior required and forbidden edges are resolved, and
         candidate pairs are screened according to the specified search space.
       - If ``expert_knowledge`` is ``None``, the full complete graph (all directed variable pairs)
         is used as the candidate search space.
    2. **Weight Upper Bound (Big-M) Estimation**:
       - Computes unconstrained least-squares regression coefficients (:math:`\\text{OLS}`) for each variable
         given candidate parents to estimate global Big-M bounds :math:`M`, used to couple continuous
         structural weights with binary edge selection decisions.
    3. **MILP Formulation Assembly**:
       - Constructs objective vector ``obj_coefficients``, integrality array, bounds vector, and sparse
       - linear constraint matrix ``constraint_matrix`` enforcing tournament ordering, Big-M weight coupling, and
         acyclicity layer ordering.
    4. **Global Exact Solution**:
       - Solves the resulting mixed-integer linear program globally via ``scipy.optimize.milp``.
    5. **Graph Extraction**:
       - Reconstructs optimal directed acyclic graph (:class:`~pgmpy.base.DAG`) from the active edge indicators.

    Parameters
    ----------
    l_penalty : float, default=0.1
        Regularization hyperparameter (:math:`\\lambda`) for the :math:`L_0` penalty. Controls edge
        sparsity: higher values enforce sparser learned graphs by increasing the cost of adding each edge.

    return_type : {"dag", "pdag", "cpdag"}, default="dag"
        The graph representation type to return for ``causal_graph_``. If ``"pdag"`` or ``"cpdag"``,
        the learned DAG is converted to a PDAG (essential graph) via ``.to_pdag()``.

    expert_knowledge : ExpertKnowledge or None, default=None
        Prior expert knowledge specifying edge constraints (required edges, forbidden edges, or custom
        candidate search space). If ``None``, the full complete graph is used as the search space.

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
        l_penalty: float = 0.1,
        return_type: str = "dag",
        expert_knowledge: ExpertKnowledge | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
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
        if get_dataset_type(X) != "continuous":
            raise ValueError("ILPSearch requires continuous (numeric) variables; got non-continuous data.")

        variable_map = {name: idx for idx, name in enumerate(self.feature_names_in_)}

        # Step 1: Superstructure & Expert Knowledge Resolution
        if self.expert_knowledge is not None:
            ek = cast(ExpertKnowledge, clone(self.expert_knowledge)).fit(X)
            required_edges = set(ek.required_edges_)
            forbidden_edges = set(ek.forbidden_edges_)
            search_space = set(ek.search_space_) if ek.search_space_ else None
        else:
            required_edges = set()
            forbidden_edges = set()
            search_space = None

        # Build candidate directed edges directly as variable index pairs (j, k)
        directed_edges = []
        for j, u in enumerate(self.feature_names_in_):
            for k, v in enumerate(self.feature_names_in_):
                if j == k:
                    continue
                # Skip edges explicitly marked as forbidden by prior knowledge
                if (u, v) in forbidden_edges:
                    continue
                # If a restricted search space is specified, ensure the edge is allowed or required
                if search_space is not None and (u, v) not in search_space and (u, v) not in required_edges:
                    continue
                directed_edges.append((j, k))

        num_directed_edges = len(directed_edges)
        directed_edge_index = {edge: i for i, edge in enumerate(directed_edges)}

        # Step 2: Big-M Upper Bound Estimation
        # Big-M is a large constant in integer programming used to link continuous edge weights (beta)
        # with binary edge choices (g). If an edge is off (g=0), Big-M forces its weight to 0.
        # If an edge is on (g=1), its weight is allowed to range freely between -M and +M.
        X_mat_raw = X.to_numpy(dtype=float)
        X_mat = X_mat_raw - X_mat_raw.mean(axis=0)

        max_ols_weight = 0.0
        for k in range(self.n_features_in_):
            parents = [j for (j, target) in directed_edges if target == k]
            if parents:
                X_parents = X_mat[:, parents]
                y = X_mat[:, k]
                beta_ols, _, _, _ = np.linalg.lstsq(X_parents, y, rcond=None)
                max_ols_weight = max(max_ols_weight, float(np.max(np.abs(beta_ols))))
        big_m_weight_bound = max(2.0 * max_ols_weight, 10.0)

        # Step 3: Decision Variables Setup
        # Vector layout: [z (orientation binaries), beta (edge weights), g (active edges), psi (node layers)]
        offset_orientation_z = 0
        offset_weight_beta = offset_orientation_z + num_directed_edges
        offset_active_g = offset_weight_beta + num_directed_edges
        offset_layer_psi = offset_active_g + num_directed_edges
        n_solver_vars = offset_layer_psi + self.n_features_in_

        # Step 4: Objective Function Setup
        # For each candidate edge (j -> k), pre-compute how much parent j reduces the error (RSS) of child k.
        # The solver minimizes: -(error reduction) + (penalty per edge).
        # An edge is selected only if its error reduction outweighs the edge penalty (l_penalty).
        obj_coefficients = np.zeros(n_solver_vars)
        n_samples = len(X_mat)

        for idx, (j, k) in enumerate(directed_edges):
            y = X_mat[:, k]
            rss_empty = float(np.sum(y**2))
            X_j = X_mat[:, [j]]
            beta_ols, _, _, _ = np.linalg.lstsq(X_j, y, rcond=None)
            rss_with_parent = float(np.sum((y - X_j @ beta_ols) ** 2))
            marginal_score_improvement = (rss_empty - rss_with_parent) / float(n_samples) if n_samples > 0 else 0.0

            active_var_idx = offset_active_g + idx
            obj_coefficients[active_var_idx] = -marginal_score_improvement + self.l_penalty

        # Variable Types: 1 for binary integers (z, g), 0 for continuous variables (beta, psi)
        integrality = np.zeros(n_solver_vars)
        integrality[offset_orientation_z : offset_orientation_z + num_directed_edges] = 1
        integrality[offset_active_g : offset_active_g + num_directed_edges] = 1

        # Variable Bounds
        lb = np.zeros(n_solver_vars)
        ub = np.zeros(n_solver_vars)

        # Orientation variables z_jk in [0, 1] (binary direction indicators)
        lb[offset_orientation_z : offset_orientation_z + num_directed_edges] = 0
        ub[offset_orientation_z : offset_orientation_z + num_directed_edges] = 1

        # Structural regression weights beta_jk in [-M, M] (bounded by Big-M)
        lb[offset_weight_beta : offset_weight_beta + num_directed_edges] = -big_m_weight_bound
        ub[offset_weight_beta : offset_weight_beta + num_directed_edges] = big_m_weight_bound

        # Edge activation variables g_jk in [0, 1] (binary edge selection)
        lb[offset_active_g : offset_active_g + num_directed_edges] = 0
        ub[offset_active_g : offset_active_g + num_directed_edges] = 1

        # Node layer potential variables psi_j in [1, n_features] (topological depth)
        lb[offset_layer_psi : offset_layer_psi + self.n_features_in_] = 1
        ub[offset_layer_psi : offset_layer_psi + self.n_features_in_] = self.n_features_in_

        # Enforce expert knowledge constraints by fixing decision variable bounds:
        # - Required edges: lower bound set to 1 (forces edge activation)
        for u, v in required_edges:
            if (variable_map[u], variable_map[v]) in directed_edge_index:
                idx = directed_edge_index[(variable_map[u], variable_map[v])]
                lb[offset_orientation_z + idx] = 1
                lb[offset_active_g + idx] = 1

        # - Forbidden edges: upper bound set to 0 (prohibits edge activation)
        for u, v in forbidden_edges:
            if (variable_map[u], variable_map[v]) in directed_edge_index:
                idx = directed_edge_index[(variable_map[u], variable_map[v])]
                ub[offset_orientation_z + idx] = 0
                ub[offset_active_g + idx] = 0

        var_bounds = Bounds(cast(Any, lb), cast(Any, ub))

        # Step 5: Build Constraints Sparse Matrix constraint_matrix
        constraint_rows = []
        constraint_lb = []
        constraint_ub = []

        # (A) Tournament Ordering Constraint: z_jk + z_kj = 1
        # For every connected pair of variables, exactly one orientation is allowed in the global ordering.
        seen_pairs = set()

        for j, k in directed_edges:
            pair = tuple(sorted((j, k)))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                row = np.zeros(n_solver_vars)
                row[offset_orientation_z + directed_edge_index[(j, k)]] = 1
                if (k, j) in directed_edge_index:
                    row[offset_orientation_z + directed_edge_index[(k, j)]] = 1
                constraint_rows.append(row)
                constraint_lb.append(1.0)
                constraint_ub.append(1.0)

        # (B) Big-M Weight Coupling Constraints: |beta_jk| <= M * g_jk and g_jk <= z_jk
        # Forces beta_jk = 0 when edge is inactive (g_jk = 0), and ensures edge j -> k
        # can only be active if it matches the topological pairwise order (z_jk = 1).
        for j, k in directed_edges:
            idx = directed_edge_index[(j, k)]
            act_idx = offset_active_g + idx
            beta_idx = offset_weight_beta + idx

            # Upper bound: beta_jk - M * g_jk <= 0 (beta_jk <= M * g_jk)
            row1 = np.zeros(n_solver_vars)
            row1[beta_idx] = 1
            row1[act_idx] = -big_m_weight_bound
            constraint_rows.append(row1)
            constraint_lb.append(-np.inf)
            constraint_ub.append(0.0)

            # Lower bound: -beta_jk - M * g_jk <= 0 (beta_jk >= -M * g_jk)
            row2 = np.zeros(n_solver_vars)
            row2[beta_idx] = -1
            row2[act_idx] = -big_m_weight_bound
            constraint_rows.append(row2)
            constraint_lb.append(-np.inf)
            constraint_ub.append(0.0)

            # Orientation link: g_jk - z_jk <= 0 (g_jk <= z_jk)
            row_link = np.zeros(n_solver_vars)
            row_link[offset_active_g + idx] = 1
            row_link[offset_orientation_z + idx] = -1
            constraint_rows.append(row_link)
            constraint_lb.append(-np.inf)
            constraint_ub.append(0.0)

        # (C) Layer Acyclicity Constraint: z_jk - (m - 1) * z_kj - psi_k + psi_j <= 0
        # Guarantees that if edge j -> k is oriented (z_jk = 1), psi_k >= psi_j + 1, preventing directed cycles.
        for j, k in directed_edges:
            row = np.zeros(n_solver_vars)
            row[offset_orientation_z + directed_edge_index[(j, k)]] = 1
            if (k, j) in directed_edge_index:
                row[offset_orientation_z + directed_edge_index[(k, j)]] = -(self.n_features_in_ - 1)
            row[offset_layer_psi + j] = 1
            row[offset_layer_psi + k] = -1
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

        # Step 7: Extract Learned DAG
        sol_active_edges = res.x[offset_active_g : offset_active_g + num_directed_edges]

        dag = DAG()
        dag.add_nodes_from(self.feature_names_in_)

        for idx, (j, k) in enumerate(directed_edges):
            if sol_active_edges[idx] > 0.5:
                dag.add_edge(self.feature_names_in_[j], self.feature_names_in_[k])

        if self.return_type in ("pdag", "cpdag"):
            self.causal_graph_ = dag.to_pdag()
        else:
            self.causal_graph_ = dag

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.feature_names_in_)
        )
        return self
