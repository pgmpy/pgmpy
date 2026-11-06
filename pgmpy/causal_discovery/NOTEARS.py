# The continuous solver in this module is adapted from notears/linear.py in
# https://github.com/xunzheng/notears (Apache License 2.0). Original work:
# Zheng, Aragam, Ravikumar and Xing, NeurIPS 2018. This file was modified for
# pgmpy's BaseCausalDiscovery API, DAG output, and input validation.
# http://www.apache.org/licenses/LICENSE-2.0

import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery


class NOTEARS(BaseCausalDiscovery):
    """
    Continuous optimization for DAG structure learning (NOTEARS) :cite:p:`zheng_2018`.

    NOTEARS casts DAG learning as a constrained continuous program: a least-squares,
    logistic, or Poisson loss plus L1 regularization, subject to the acyclicity
    constraint ``h(W) = tr(exp(W ◦ W)) - d = 0``. The equality is handled with an
    augmented Lagrangian; each inner step is L-BFGS-B on a nonnegative split
    ``W = W+ - W-``. After optimization, entries with absolute weight below
    ``w_threshold`` are dropped. Remaining edges are inserted in decreasing
    absolute weight so ``causal_graph_`` is a :class:`~pgmpy.base.DAG`.

    For ``loss_type='l2'`` the columns of ``X`` are mean-centered, matching
    ``xunzheng/notears`` (Apache-2.0). Columns are not standardized. The
    estimator tags describe this continuous linear case; ``logistic`` and
    ``poisson`` are extra numeric likelihoods on 0/1 and count data.

    Parameters
    ----------
    lambda1 : float, default=0.1
        L1 penalty on edge weights.

    loss_type : {'l2', 'logistic', 'poisson'}, default='l2'
        Data likelihood. ``l2`` is linear SEM / continuous data, ``logistic`` is
        binary 0/1 data, ``poisson`` is non-negative counts.

    max_iter : int, default=100
        Maximum number of augmented-Lagrangian dual steps.

    h_tol : float, default=1e-8
        Stop when the acyclicity residual is at most this value.

    rho_max : float, default=1e16
        Stop when the penalty parameter reaches this value.

    w_threshold : float, default=0.3
        Drop edges whose absolute weight is below this value. A value of 0
        keeps the raw solver output, which can contain 2-cycles; weaker
        cyclic edges are then dropped.

    show_progress : bool, default=True
        If True and global progress is enabled, show a progress bar.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned directed acyclic graph. Edge attribute ``weight`` stores the
        estimated coefficient.

    adjacency_matrix_ : pd.DataFrame
        Binary adjacency matrix of ``causal_graph_``.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import NOTEARS
    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(size=200)
    >>> y = 1.5 * x + rng.normal(scale=0.3, size=200)
    >>> data = pd.DataFrame({"X": x, "Y": y})
    >>> est = NOTEARS(show_progress=False).fit(data)
    >>> set(est.causal_graph_.edges())
    {('X', 'Y')}

    References
    ----------
    - :cite:p:`zheng_2018`
    - https://github.com/xunzheng/notears
    """

    _tags = {
        "name": "notears",
        "data_types": ["continuous"],
        "identifiable_graph": "dag",
        "requires_target": False,
        "capability:multivariate": True,
        "capability:expert_knowledge": [],
        "assumption:causal_sufficiency": True,
        "assumption:acyclicity": True,
        "assumption:faithfulness": False,
        "assumption:linearity": True,
        "assumption:additive_noise": True,
        "assumption:gaussian_noise": False,
        "assumption:non_gaussian_noise": False,
        "assumption:low_noise": False,
    }

    def __init__(
        self,
        lambda1: float = 0.1,
        loss_type: str = "l2",
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
        show_progress: bool = True,
    ) -> None:
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.show_progress = show_progress

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    def _fit(self, X: pd.DataFrame) -> "NOTEARS":
        """
        The fitting procedure for the NOTEARS algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.NOTEARS
            Returns the instance with the fitted attributes.
        """
        if self.loss_type not in {"l2", "logistic", "poisson"}:
            raise ValueError(f"loss_type must be one of: l2, logistic, poisson. Got: {self.loss_type}")
        if self.lambda1 < 0:
            raise ValueError(f"lambda1 must be non-negative. Got: {self.lambda1}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be a positive integer. Got: {self.max_iter}")
        if self.w_threshold < 0:
            raise ValueError(f"w_threshold must be non-negative. Got: {self.w_threshold}")

        data = np.asarray(X.to_numpy(dtype=float), dtype=float)
        n_samples, n_nodes = data.shape
        nodes = list(self.feature_names_in_)

        if self.loss_type == "l2":
            data = data - np.mean(data, axis=0, keepdims=True)
        elif self.loss_type == "logistic":
            is_binary = np.logical_or(np.isclose(data, 0.0), np.isclose(data, 1.0))
            if not np.all(is_binary):
                raise ValueError("For loss_type='logistic', all values must be binary (0/1).")
        elif np.any(data < 0):
            raise ValueError("For loss_type='poisson', all values must be non-negative.")

        def _loss(adjacency):
            scores = data @ adjacency
            if self.loss_type == "l2":
                residual = data - scores
                loss = 0.5 / n_samples * (residual**2).sum()
                loss_jac = -1.0 / n_samples * data.T @ residual
            elif self.loss_type == "logistic":
                loss = 1.0 / n_samples * (np.logaddexp(0.0, scores) - data * scores).sum()
                loss_jac = 1.0 / n_samples * data.T @ (sigmoid(scores) - data)
            else:
                exp_scores = np.exp(scores)
                loss = 1.0 / n_samples * (exp_scores - data * scores).sum()
                loss_jac = 1.0 / n_samples * data.T @ (exp_scores - data)
            return loss, loss_jac

        def _acyclicity(adjacency):
            matrix_exp = slin.expm(adjacency * adjacency)
            penalty = np.trace(matrix_exp) - n_nodes
            jac = matrix_exp.T * adjacency * 2
            return penalty, jac

        def _to_adjacency(doubled):
            n_entries = n_nodes * n_nodes
            return (doubled[:n_entries] - doubled[n_entries:]).reshape((n_nodes, n_nodes))

        def _objective(doubled):
            adjacency = _to_adjacency(doubled)
            loss, loss_jac = _loss(adjacency)
            penalty, penalty_jac = _acyclicity(adjacency)
            objective = loss + 0.5 * rho * penalty * penalty + alpha * penalty + self.lambda1 * doubled.sum()
            smooth_jac = loss_jac + (rho * penalty + alpha) * penalty_jac
            return objective, np.concatenate((smooth_jac + self.lambda1, -smooth_jac + self.lambda1), axis=None)

        doubled = np.zeros(2 * n_nodes * n_nodes)
        rho = 1.0
        alpha = 0.0
        penalty = np.inf
        bounds = [
            (0.0, 0.0) if i == j else (0.0, None) for _ in range(2) for i in range(n_nodes) for j in range(n_nodes)
        ]
        iterations = (
            trange(int(self.max_iter)) if self.show_progress and config.SHOW_PROGRESS else range(int(self.max_iter))
        )

        # Augmented Lagrangian outer loop from Zheng et al. 2018 / xunzheng/notears.
        for _ in iterations:
            doubled_new, penalty_new = None, None
            while rho < self.rho_max:
                result = sopt.minimize(_objective, doubled, method="L-BFGS-B", jac=True, bounds=bounds)
                doubled_new = result.x
                penalty_new, _ = _acyclicity(_to_adjacency(doubled_new))
                if penalty_new > 0.25 * penalty:
                    rho *= 10
                else:
                    break
            doubled, penalty = doubled_new, penalty_new
            alpha += rho * penalty
            if penalty <= self.h_tol or rho >= self.rho_max:
                break

        adjacency = _to_adjacency(doubled)
        adjacency[np.abs(adjacency) < self.w_threshold] = 0.0
        ranked = []
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                weight = float(adjacency[i, j])
                if weight != 0.0:
                    ranked.append((abs(weight), i, j, source, target, weight))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))

        # Official notears_linear returns W, which need not be a DAG at w_threshold=0.
        scratch = nx.DiGraph()
        scratch.add_nodes_from(nodes)
        edges = []
        weights = []
        for _, _, _, source, target, weight in ranked:
            scratch.add_edge(source, target)
            if nx.is_directed_acyclic_graph(scratch):
                edges.append((source, target))
                weights.append(weight)
            else:
                scratch.remove_edge(source, target)

        self.causal_graph_ = DAG()
        self.causal_graph_.add_nodes_from(nodes)
        if edges:
            self.causal_graph_.add_edges_from(edges, weights=weights)
        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(encoding="binary", nodelist=nodes)
        return self
