from functools import partial

import networkx as nx
import numpy as np
import scipy.optimize as sopt
from scipy.special import expit
from skbase.utils.dependencies import _safe_import
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.causal_discovery.ExpertKnowledge import ExpertKnowledge
from pgmpy.global_vars import logger
from pgmpy.utils import compat_fns

torch = _safe_import("torch")


class NOTEARS(_BaseCausalDiscovery):
    """
    NOTEARS structure learning estimator.

    Learns a DAG from continuous, binary, or count data by solving a continuous
    optimization problem with an acyclicity constraint based on the matrix exponential.

    Parameters
    ----------
    lambda1 : float, default=0.1
        L1 regularization coefficient on edge strengths.

    loss_type : str, default='l2'
        Loss function to use. One of 'l2', 'logistic', or 'poisson'.

    max_iter : int, default=20
        Maximum number of outer augmented Lagrangian iterations.

    h_tol : float, default=1e-8
        Stops when acyclicity violation <= h_tol.

    rho_max : float, default=1e16
        Maximum penalty parameter for augmented Lagrangian.

    w_threshold : float, default=0.3
        Edges with absolute weight below this are dropped.

    expert_knowledge : ExpertKnowledge or None, default=None
        Prior knowledge on required/forbidden edges and temporal ordering.

    show_progress : bool, default=True
        If True and global progress is enabled, shows a progress bar.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned directed acyclic graph.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix of the learned causal graph.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : np.ndarray
        Feature names seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import NOTEARS
    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(size=500)
    >>> y = 1.5 * x + rng.normal(scale=0.3, size=500)
    >>> data = pd.DataFrame({"X": x, "Y": y})
    >>> est = NOTEARS(lambda1=0.1, loss_type="l2")
    >>> est.fit(data)
    >>> est.causal_graph_.edges()

    References
    ----------
    .. [1] Zheng, X., Aragam, B., Ravikumar, P., and Xing, E. P. (2018).
           DAGs with NO TEARS: Continuous Optimization for Structure Learning.
           Advances in Neural Information Processing Systems 31.
    """

    _required_penalty_weight = 10.0

    def __init__(
        self,
        lambda1=0.1,
        loss_type="l2",
        max_iter=20,
        h_tol=1e-8,
        rho_max=1e16,
        w_threshold=0.3,
        expert_knowledge=None,
        show_progress=True,
    ):
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.expert_knowledge = expert_knowledge
        self.show_progress = show_progress

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    @staticmethod
    def _doubled_to_adjacency(adjacency_doubled, n_nodes):
        n_entries = n_nodes * n_nodes
        return (adjacency_doubled[:n_entries] - adjacency_doubled[n_entries:]).reshape((n_nodes, n_nodes))

    def _validate_data_for_loss(self, X):
        data_np = X.to_numpy(dtype=float)

        if np.isnan(data_np).any():
            raise ValueError("NOTEARS does not support missing values.")

        if self.loss_type == "logistic":
            is_binary = np.logical_or(np.isclose(data_np, 0.0), np.isclose(data_np, 1.0))
            if not np.all(is_binary):
                raise ValueError("For loss_type='logistic', all values must be binary (0/1).")
        elif self.loss_type == "poisson":
            if np.any(data_np < 0):
                raise ValueError("For loss_type='poisson', all values must be non-negative.")

        return data_np

    def _constraint_grad(self, adjacency_matrix, compute_jac=True):
        matrix_exp = compat_fns.matrix_exp(adjacency_matrix * adjacency_matrix)
        if isinstance(adjacency_matrix, np.ndarray):
            penalty = np.trace(matrix_exp) - adjacency_matrix.shape[0]
        else:
            penalty = torch.trace(matrix_exp) - adjacency_matrix.shape[0]
        jac = (matrix_exp.T * adjacency_matrix * 2) if compute_jac else None
        return penalty, jac

    def _required_penalty_gradient(self, adjacency_strength, required_mask, weight_threshold, compute_jac=True):
        deficit = weight_threshold - adjacency_strength
        active = deficit > 0

        if isinstance(adjacency_strength, np.ndarray):
            penalty_mat = np.where(active, deficit * deficit, 0.0) * required_mask
            jac = np.where(active, -2.0 * deficit, 0.0) * required_mask if compute_jac else None
            return np.sum(penalty_mat), jac
        else:
            penalty_mat = torch.where(active, deficit * deficit, torch.zeros_like(deficit)) * required_mask
            jac = (
                (torch.where(active, -2.0 * deficit, torch.zeros_like(deficit)) * required_mask)
                if compute_jac
                else None
            )
            return torch.sum(penalty_mat), jac

    def _required_edge_min_strength(self):
        return max(self.w_threshold, self.lambda1 / (2 * self._required_penalty_weight) + 1e-6)

    @staticmethod
    def _validate_expert_knowledge_nodes(expert_knowledge, node_to_index):
        for edge_set in (expert_knowledge.required_edges, expert_knowledge.forbidden_edges):
            for u, v in edge_set:
                if (u not in node_to_index) or (v not in node_to_index):
                    raise ValueError(f"Expert knowledge edge ({u}, {v}) refers to node(s) not present in the data columns.")

    def _loss_grad(self, data, adjacency_matrix, backend):
        scores = data @ adjacency_matrix
        n_samples = data.shape[0]

        if self.loss_type == "l2":
            residual = data - scores
            loss = 0.5 / n_samples * backend.sum(residual**2)
            loss_jac = -(data.T @ residual) / n_samples
            return loss, loss_jac

        if self.loss_type == "logistic":
            if isinstance(data, np.ndarray):
                loss = 1.0 / n_samples * (np.logaddexp(0.0, scores) - data * scores).sum()
                probs = expit(scores)
            else:
                loss = 1.0 / n_samples * (torch.logaddexp(torch.zeros_like(scores), scores) - data * scores).sum()
                probs = torch.sigmoid(scores)
            loss_jac = 1.0 / n_samples * data.T @ (probs - data)
            return loss, loss_jac

        # poisson
        exp_scores = backend.exp(scores)
        loss = 1.0 / n_samples * backend.sum(exp_scores - data * scores)
        loss_jac = 1.0 / n_samples * data.T @ (exp_scores - data)
        return loss, loss_jac

    def _objective_numpy(
        self,
        adjacency_doubled,
        alpha,
        rho,
        data,
        required_mask,
        backend,
    ):
        n_nodes = data.shape[1]
        n_entries = n_nodes * n_nodes
        w_plus = adjacency_doubled[:n_entries].reshape((n_nodes, n_nodes))
        w_minus = adjacency_doubled[n_entries:].reshape((n_nodes, n_nodes))

        adjacency_matrix = w_plus - w_minus
        adjacency_strength = w_plus + w_minus

        loss, loss_jac = self._loss_grad(data, adjacency_matrix, backend)
        acyclic_penalty, acyclic_jac = self._constraint_grad(adjacency_matrix)
        required_threshold = self._required_edge_min_strength()
        required_penalty, required_jac = self._required_penalty_gradient(
            adjacency_strength, required_mask, required_threshold
        )

        objective = (
            loss
            + 0.5 * rho * acyclic_penalty * acyclic_penalty
            + alpha * acyclic_penalty
            + self.lambda1 * adjacency_strength.sum()
            + self._required_penalty_weight * required_penalty
        )

        adjacency_jac = loss_jac + (rho * acyclic_penalty + alpha) * acyclic_jac
        jac_plus = adjacency_jac + self.lambda1 + self._required_penalty_weight * required_jac
        jac_minus = -adjacency_jac + self.lambda1 + self._required_penalty_weight * required_jac
        jac = compat_fns.concatenate(jac_plus.ravel(), jac_minus.ravel())
        return objective, jac

    def _objective_torch(
        self,
        adjacency_doubled,
        alpha,
        rho,
        data,
        required_mask,
        hard_mask,
        backend,
    ):
        n_nodes = data.shape[1]
        n_entries = n_nodes * n_nodes
        w_plus = adjacency_doubled[:n_entries].reshape((n_nodes, n_nodes))
        w_minus = adjacency_doubled[n_entries:].reshape((n_nodes, n_nodes))

        w_plus = torch.clamp(w_plus, min=0.0)
        w_minus = torch.clamp(w_minus, min=0.0)
        zero_matrix = torch.zeros_like(w_plus)
        w_plus = torch.where(hard_mask, zero_matrix, w_plus)
        w_minus = torch.where(hard_mask, zero_matrix, w_minus)

        adjacency_matrix = w_plus - w_minus
        adjacency_strength = w_plus + w_minus
        loss, _ = self._loss_grad(data, adjacency_matrix, backend)
        acyclic_penalty, _ = self._constraint_grad(adjacency_matrix, compute_jac=False)
        required_threshold = self._required_edge_min_strength()
        required_penalty, _ = self._required_penalty_gradient(
            adjacency_strength, required_mask, required_threshold, compute_jac=False
        )

        return (
            loss
            + 0.5 * rho * acyclic_penalty * acyclic_penalty
            + alpha * acyclic_penalty
            + self.lambda1 * torch.sum(adjacency_strength)
            + self._required_penalty_weight * required_penalty
        )

    @staticmethod
    def _project_torch_doubled(adjacency_doubled, n_nodes, hard_mask):
        with torch.no_grad():
            n_entries = n_nodes * n_nodes
            w_plus = adjacency_doubled[:n_entries].reshape((n_nodes, n_nodes))
            w_minus = adjacency_doubled[n_entries:].reshape((n_nodes, n_nodes))

            w_plus.clamp_(min=0.0)
            w_minus.clamp_(min=0.0)
            w_plus[hard_mask] = 0.0
            w_minus[hard_mask] = 0.0

    def _run_inner_torch(
        self,
        adjacency_doubled,
        alpha,
        rho,
        data,
        required_mask,
        hard_mask,
        backend,
    ):
        adjacency_param = torch.nn.Parameter(adjacency_doubled.detach().clone())
        optimizer = torch.optim.LBFGS([adjacency_param], max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            objective = self._objective_torch(
                adjacency_param,
                alpha=alpha,
                rho=rho,
                data=data,
                required_mask=required_mask,
                hard_mask=hard_mask,
                backend=backend,
            )
            objective.backward()
            return objective

        optimizer.step(closure)
        self._project_torch_doubled(adjacency_param, data.shape[1], hard_mask)
        return adjacency_param.detach()

    @staticmethod
    def _standardize_l2(data):
        if isinstance(data, np.ndarray):
            means = data.mean(axis=0, keepdims=True)
            stds = data.std(axis=0, keepdims=True)
            stds[stds == 0] = 1.0
            return (data - means) / stds

        means = data.mean(dim=0, keepdim=True)
        stds = data.std(dim=0, keepdim=True, unbiased=False)
        stds = torch.where(stds == 0, torch.ones_like(stds), stds)
        return (data - means) / stds

    def _fit(self, X):
        """
        Run the NOTEARS optimization on data X.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : NOTEARS
            Returns the instance with fitted attributes.
        """
        if self.loss_type not in {"l2", "logistic", "poisson"}:
            raise ValueError(f"loss_type must be one of: l2, logistic, poisson. Got: {self.loss_type}")
        if self.lambda1 < 0:
            raise ValueError(f"lambda1 must be non-negative. Got: {self.lambda1}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be a positive integer. Got: {self.max_iter}")
        if self.w_threshold < 0:
            raise ValueError(f"w_threshold must be non-negative. Got: {self.w_threshold}")

        data_np = self._validate_data_for_loss(X)
        backend = compat_fns.get_compute_backend()

        nodes = list(X.columns)
        n_nodes = len(nodes)
        node_to_index = {node: idx for idx, node in enumerate(nodes)}

        expert_knowledge = ExpertKnowledge() if self.expert_knowledge is None else self.expert_knowledge
        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(nodes)
        expert_knowledge._orient_temporal_forbidden_edges(DAG(), only_edges=False)
        self._validate_expert_knowledge_nodes(expert_knowledge, node_to_index)

        forbidden_mask_np = np.zeros((n_nodes, n_nodes), dtype=bool)
        required_mask_np = np.zeros((n_nodes, n_nodes), dtype=float)

        for u, v in expert_knowledge.required_edges:
            if (u, v) in expert_knowledge.forbidden_edges:
                raise ValueError(f"Expert knowledge conflict: Edge ({u}, {v}) is both required and forbidden.")
            if u == v:
                raise ValueError(f"Expert knowledge conflict: Self-loop ({u}, {u}) cannot be required.")

        for u, v in expert_knowledge.forbidden_edges:
            if (u in node_to_index) and (v in node_to_index):
                forbidden_mask_np[node_to_index[u], node_to_index[v]] = True
        np.fill_diagonal(forbidden_mask_np, True)

        for u, v in expert_knowledge.required_edges:
            if (u in node_to_index) and (v in node_to_index):
                required_mask_np[node_to_index[u], node_to_index[v]] = 1.0

        if backend == np:
            data = np.asarray(data_np, dtype=config.get_dtype())
            required_mask = np.asarray(required_mask_np, dtype=config.get_dtype())
            adjacency_doubled = np.zeros(2 * n_nodes * n_nodes, dtype=config.get_dtype())
        else:
            data = torch.tensor(data_np, dtype=config.get_dtype(), device=config.get_device())
            required_mask = torch.tensor(required_mask_np, dtype=config.get_dtype(), device=config.get_device())
            adjacency_doubled = torch.zeros(
                2 * n_nodes * n_nodes,
                dtype=config.get_dtype(),
                device=config.get_device(),
            )
            hard_mask = torch.tensor(forbidden_mask_np, dtype=torch.bool, device=data.device)

        if self.loss_type == "l2":
            data = self._standardize_l2(data)

        if self.show_progress and config.SHOW_PROGRESS:
            iteration = trange(int(self.max_iter))
        else:
            iteration = range(int(self.max_iter))

        rho = 1.0
        alpha = 0.0
        h = np.inf

        if backend == np:
            bounds = []
            for _ in range(2):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if forbidden_mask_np[i, j]:
                            bounds.append((0.0, 0.0))
                        else:
                            bounds.append((0.0, None))

            for iter_idx in iteration:
                adjacency_new = adjacency_doubled
                h_new = h
                while rho < self.rho_max:
                    objective = partial(
                        self._objective_numpy,
                        alpha=alpha,
                        rho=rho,
                        data=data,
                        required_mask=required_mask,
                        backend=backend,
                    )
                    result = sopt.minimize(
                        objective,
                        adjacency_doubled,
                        method="L-BFGS-B",
                        jac=True,
                        bounds=bounds,
                    )
                    candidate = result.x
                    h_candidate, _ = self._constraint_grad(
                        self._doubled_to_adjacency(candidate, n_nodes), compute_jac=False
                    )
                    h_new = float(h_candidate)

                    if np.isfinite(h) and h_new > 0.25 * h:
                        rho *= 10
                    else:
                        adjacency_new = candidate
                        break

                adjacency_doubled = adjacency_new
                h = h_new
                alpha += rho * h
                logger.debug(
                    "NOTEARS iter=%s h=%.4e rho=%.4e alpha=%.4e",
                    iter_idx,
                    h,
                    rho,
                    alpha,
                )
                if h <= self.h_tol or rho >= self.rho_max:
                    break
        else:
            for iter_idx in iteration:
                adjacency_new = adjacency_doubled
                h_new = h
                while rho < self.rho_max:
                    candidate = self._run_inner_torch(
                        adjacency_doubled=adjacency_doubled,
                        alpha=alpha,
                        rho=rho,
                        data=data,
                        required_mask=required_mask,
                        hard_mask=hard_mask,
                        backend=backend,
                    )
                    h_candidate, _ = self._constraint_grad(
                        self._doubled_to_adjacency(candidate, n_nodes), compute_jac=False
                    )
                    h_new = float(h_candidate.detach().cpu().item())

                    if np.isfinite(h) and h_new > 0.25 * h:
                        rho *= 10
                    else:
                        adjacency_new = candidate
                        break

                adjacency_doubled = adjacency_new
                h = h_new
                alpha += rho * h
                logger.debug(
                    "NOTEARS iter=%s h=%.4e rho=%.4e alpha=%.4e",
                    iter_idx,
                    h,
                    rho,
                    alpha,
                )
                if h <= self.h_tol or rho >= self.rho_max:
                    break

        adjacency_est = self._doubled_to_adjacency(adjacency_doubled, n_nodes)
        adjacency_np = compat_fns.to_numpy(adjacency_est)
        adjacency_np[np.abs(adjacency_np) < self.w_threshold] = 0.0

        if expert_knowledge.required_edges:
            required_threshold = self._required_edge_min_strength()
            for u, v in expert_knowledge.required_edges:
                u_idx = node_to_index[u]
                v_idx = node_to_index[v]
                if adjacency_np[u_idx, v_idx] == 0.0:
                    adjacency_np[u_idx, v_idx] = required_threshold

        dag = DAG()
        dag.add_nodes_from(nodes)
        required_edges = set(expert_knowledge.required_edges)
        for u, v in required_edges:
            if nx.has_path(dag, v, u):
                raise ValueError("required_edges create a cycle in the output DAG. Please modify required_edges.")
            dag.add_edge(u, v)

        weighted_edges = [
            (nodes[i], nodes[j], abs(adjacency_np[i, j]))
            for i in range(n_nodes)
            for j in range(n_nodes)
            if (i != j) and (adjacency_np[i, j] != 0.0) and ((nodes[i], nodes[j]) not in required_edges)
        ]
        weighted_edges.sort(key=lambda edge: edge[2], reverse=True)
        for u, v, _ in weighted_edges:
            if not nx.has_path(dag, v, u):
                dag.add_edge(u, v)

        self.causal_graph_ = dag
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, weight=1, dtype="int")

        return self
