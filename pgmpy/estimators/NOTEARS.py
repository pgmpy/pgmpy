import math
from functools import partial

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import torch
from scipy.special import expit as sigmoid
from torch.optim import LBFGS
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge, StructureEstimator
from pgmpy.global_vars import logger
from pgmpy.utils import compat_fns


class NOTEARS(StructureEstimator):
    """
    Implementation of the NOTEARS structure learning algorithm.

    NOTEARS is a structure learning algorithm that works by converting the discrete score-based
    DAG learning problem into a continuous optimisation problem.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    References
    ----------
    Xun Zheng, Bryon Aragam, Pradeep Ravikumar, and Eric P. Xing. 2018. DAGs with NO TEARS: continuous
    optimization for structure learning. In Proceedings of the 32nd International Conference on Neural
    Information Processing Systems (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 9492–9503.
    """

    def __init__(self, data, args=None, **kwargs):
        super(NOTEARS, self).__init__(data=data, **kwargs)
        self.backend = compat_fns.get_compute_backend()
        self.args = args

    def _loss_grad(self, data, adjacency_matrix, loss_type):
        """
        Calculate the value of the loss function and its gradient.

        Parameters
        ----------
        data: numpy.ndarray
            The data passed for DAG estimation.

        adjacency_matrix: numpy.ndarray
            The weighted adjacency matrix for the DAG.

        loss_type: str (l2 | logistic | poisson)
            The method to use for computing the loss.

        Returns
        -------
        loss: float
            The value of the loss function for given data.

        loss_jac: numpy.ndarray
            The gradient of the loss function for given data.
        """
        if self.backend == torch:
            adjacency_matrix = adjacency_matrix.double()

        M = data @ adjacency_matrix
        n_variables = adjacency_matrix.shape[0]

        if loss_type == "l2":
            R = data - M
            loss = (0.5 * (R**2).sum()) / (data.shape[0])
            loss_jac = -(data.T @ R) / (data.shape[0])

        elif loss_type == "logistic":
            loss = (1.0 / data.shape[0]) * (
                self.backend.logaddexp(0, M) - data * M
            ).sum()
            loss_jac = (1.0 / data.shape[0]) * data.T @ (sigmoid(M) - data)

        elif loss_type == "poisson":
            S = self.backend.exp(M)
            loss = (1.0 / data.shape[0]) * (S - data * M).sum()
            loss_jac = (1.0 / data.shape[0]) * data.T @ (S - data)

        else:
            raise ValueError(
                "Unknown `loss_type`. Only `l2`, `logistic`, and `poisson` are supported."
            )

        return loss, loss_jac

    def _constraint_grad(self, adjacency_matrix):
        """
        Calculate the value of the acyclicity constraint and its gradient.

        The acyclicity constraint is given by:

        .. math:: h(W) = \textit{tr}(e^{(W*W)}) - d

        where :math:`W` is the weighted adjacency matrix and :math:`d` is the
        number of nodes in graph.

        Parameters
        ----------
        adjacency_matrix: 2D array
            The weighted adjacency matrix of shape d x d.

        Returns
        -------
        acyclic_penalty: float
            The penalty value for the given `adjacency_matrix`.

        acyclic_jac: numpy.ndarray
            The gradient of the penalty at the `adjacency_matrix`.
        """
        E = compat_fns.matrix_exp(adjacency_matrix * adjacency_matrix)
        acyclic_penalty = self.backend.trace(E) - adjacency_matrix.shape[0]
        acyclic_jac = E.T * adjacency_matrix * 2
        return acyclic_penalty, acyclic_jac

    def _adj(self, adjacency_matrix_doubled):
        """
        Convert doubled array ([2 d^2] array) back to original variables ([d, d] matrix).

        Parameters
        ----------
        adjacency_matrix_doubled: 2D array
            The doubled adjacency matrix that need to be converted back to an adjacency matrix.

        Returns
        -------
        adjacency_matrix: numpy.ndarray
            The weighted matrix in original dimensions (n*n)
        """
        dim = np.sqrt(adjacency_matrix_doubled.shape[0] / 2).astype(int)
        return (
            adjacency_matrix_doubled[: dim * dim]
            - adjacency_matrix_doubled[dim * dim :]
        ).reshape([dim, dim])

    def _forbidden_penalty_gradient(self, adjacency_matrix, forbidden_mask):
        """
        Evaluate value and gradient for masking adjacency matrix using forbidden edge data.

        Parameters
        ----------
        adjacency_matrix: 2D array
            The weighted adjacency matrix.

        forbidden_mask: 2D array
            A binary mask representing the forbidden edges. A value of 1 in the
            mask represents that the corresponding edge is forbidden.

        Returns
        -------
        loss: float
            The value of the objective function for given parameters.

        grad: numpy.ndarray
            The gradient of the objective function for given parameters.
        """
        # Given a mask M, and adjacency matrix W, the penalty is given by $ \sum_{i,j} M_{ij} W_{ij}^2 $.
        # The Jacobian is given by $ J_{ij} = 2 * W_{ij} M_{ij} M_{ij} = 2 * W_{ij} * M_{ij} $.

        penalty = self.backend.sum(
            (self.backend.abs(adjacency_matrix) * forbidden_mask) ** 2
        )
        jac = 2 * adjacency_matrix * forbidden_mask
        return penalty, jac

    def _fixed_penalty_gradient(
        self, adjacency_matrix, fixed_mask, alpha, weight_threshold
    ):
        """
        Evaluate value and gradient for masking adjacency matrix using required edge data.

        Parameters
        ----------
        adjacency_matrix: 2D array
            The weighted adjacency matrix.

        fixed_mask: 2D array
            A binary mask array representing which edges need to exist in the
            graph. A value of 1 in the mask represents that the corresponding
            edge in the adjacency matrix must exist.

        alpha: float
            TODO:

        weight_threshold: float
            The threshold for the values in adjacency_matrix above which it is
            considered to represent an edge.

        Returns
        -------
        loss: float
            The penalty for fixed edges at the given adjacency_matrix.

        grad: numpy.ndarray
            The gradient of the penalty at the given adjacency_matrix.
        """
        # Given a mask M, and adjacency matrix W, we define W^t = W if W < w_threshold else 0.
        # The penalty matrix is: $ P_{ij} e^{-(w_t - W^t_{ij} * M_{ij})^{- \alpha}} $
        # The penalty is $ \sum_{ij} P_{ij} $.
        # The Jacobian is $ J_{ij} = -\alpha M_{ij} P_{ij} (w_t -  W_{ij} M_{ij})^{- \alpha - 1} $.
        # changes?

        W_thres = self.backend.where(
            adjacency_matrix < weight_threshold, adjacency_matrix, 0
        )
        penalty_mat = self.backend.exp(
            -((weight_threshold - (W_thres * fixed_mask)) ** (-alpha))
        )
        jac = (
            -alpha
            * fixed_mask
            * penalty_mat
            * (weight_threshold - W_thres * fixed_mask) ** (-alpha - 1)
        )

        return self.backend.sum(penalty_mat), jac

    def _objective_function(
        self,
        adjacency_matrix_doubled,
        alpha,
        rho,
        lambda1,
        lambda2,
        lambda3,
        loss_type,
        data,
        forbidden_mask,
        fixed_mask,
        w_threshold,
        alpha_2,
    ):
        """
        Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array).

        Parameters
        ----------

        Returns
        -------
        obj: float
            The value of the objective function for given parameters.

        g_obj: numpy.ndarray
            The gradient of the objective function for given parameters.
        """
        adjacency_matrix_est = self._adj(adjacency_matrix_doubled)
        loss, loss_jac = self._loss_grad(data, adjacency_matrix_est, loss_type)
        acyclic_penalty, acyclic_jac = self._constraint_grad(adjacency_matrix_est)

        forb_penalty, forb_jac = self._forbidden_penalty_gradient(
            adjacency_matrix_est, forbidden_mask
        )
        fixed_penalty, fixed_jac = self._fixed_penalty_gradient(
            adjacency_matrix_est, fixed_mask, alpha_2, w_threshold
        )
        total_loss = (
            loss
            + 0.5 * rho * acyclic_penalty * acyclic_penalty
            + alpha * acyclic_penalty
            + lambda1 * adjacency_matrix_doubled.sum()
            + lambda2 * forb_penalty
            + lambda3 * fixed_penalty
        )

        if self.backend == torch:
            return total_loss

        loss_gradient = (
            loss_jac
            + (rho * acyclic_penalty + alpha) * acyclic_jac
            + lambda2 * forb_jac
            + lambda3 * fixed_jac
        )

        loss_gradient_double = compat_fns.concatenate(
            loss_gradient + lambda1 + lambda2 * forb_jac + lambda3 * fixed_jac,
            -loss_gradient + lambda1 + lambda2 * forb_jac + lambda3 * fixed_jac,
        )

        return total_loss, loss_gradient_double

    def closure(self):
        self.lbfgs.zero_grad()  # Clear previous gradients
        loss, _ = self._objective_function(*self.args)
        loss = torch.tensor(loss, dtype=config.DTYPE, requires_grad=True)
        loss.backward()
        return loss

    def estimate(
        self,
        lambda1,
        lambda2=10,
        lambda3=10,
        loss_type="l2",
        max_iter=20,
        h_tol=1e-8,
        rho_max=1e16,
        w_threshold=0.3,
        alpha_2=10,
        expert_knowledge=None,
        show_progress=True,
    ):
        """
        Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

        The objective function, consisiting of the sum of a loss function and the L1 norm,
        is minimised using the augmented Lagrangian method. The resulting weighted matrix
        is the Structural Equation Model corresponding to the data. The weighted matrix W is
        converted to a DAG object and returned.

        Parameters
        ----------
        lambda1: float
            penalty parameter for l1 norm

        loss_type: str (one of "l2", "logistic", "poisson")
            the loss function to be used in objective function

        max_iter: int
            maximum number of dual ascent steps

        h_tol: float
            exit if |h(w_est)| <= htol

        rho_max: float
            exit if rho >= rho_max

        w_threshold: float
            drop edge if |weight| < w_threshold

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            The estimated model with (local) minimum value of objective function.

        Examples
        --------
        """

        # Step 0: Initial checks and setup
        if loss_type not in ("l2", "logistic", "poisson"):
            raise ValueError(
                f"loss_type must be one of: l2, logistic, or poisson. Got: {loss_type}"
            )

        if show_progress and config.SHOW_PROGRESS:
            iteration = trange(int(max_iter))
        else:
            iteration = range(int(max_iter))

        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        expert_knowledge._orient_temporal_forbidden_edges(DAG(), only_edges=False)
        n = self.data.shape[1]
        nodes = self.data.columns.to_list()

        if config.get_backend() == "numpy":
            data = np.array(self.data.values, dtype=config.get_dtype())
        else:
            data = torch.tensor(
                self.data.values, dtype=config.get_dtype(), device=config.get_device()
            )

        # Step 1: Initialize starting estimates
        adjacency_matrix_est = self.backend.zeros(2 * n * n)
        rho = 1.0
        alpha = 0.0
        h = self.backend.inf

        # Step 1.1: Initialize a mask for forbidden edges
        forbidden_mask = self.backend.zeros((n, n))
        for u, v in expert_knowledge.forbidden_edges:
            i = nodes.index(u)
            j = nodes.index(v)
            forbidden_mask[i][j] = 1

        # Step 1.2: Initialize a mask for fixed edges
        fixed_mask = self.backend.zeros((n, n))
        for u, v in expert_knowledge.required_edges:
            i = nodes.index(u)
            j = nodes.index(v)
            fixed_mask[i][j] = 1

        bounds = [
            (0, 0) if i == j else (0, None)
            for _ in range(2)
            for i in range(n)
            for j in range(n)
        ]
        self.args = [
            alpha,
            rho,
            lambda1,
            lambda2,
            lambda3,
            loss_type,
            data,
            forbidden_mask,
            fixed_mask,
            w_threshold,
            alpha_2,
        ]

        if loss_type == "l2":
            logger.info("Normalizing data to 0 mean and unit standard deviation.")
            data = (
                data - self.backend.mean(data, axis=0, keepdims=True)
            ) / self.backend.std(data, axis=0, keepdims=True)

        # Step 2: Iterate until max_iter iterations
        if self.backend == np:
            for _ in iteration:
                adjacency_matrix_new, h_new = None, None

                # Step 2.(a): Solve for new values of W and h
                while rho < rho_max:
                    obj_fun = partial(
                        self._objective_function,
                        alpha=alpha,
                        rho=rho,
                        lambda1=lambda1,
                        lambda2=lambda2,
                        lambda3=lambda3,
                        loss_type=loss_type,
                        data=data,
                        forbidden_mask=forbidden_mask,
                        fixed_mask=fixed_mask,
                        w_threshold=w_threshold,
                        alpha_2=alpha_2,
                    )

                    if self.backend == np:
                        sol = sopt.minimize(
                            obj_fun,
                            adjacency_matrix_est,
                            method="L-BFGS-B",
                            jac=True,
                            bounds=bounds,
                        )
                        adjacency_matrix_new = sol.x
                        h_new, _ = self._constraint_grad(
                            self._adj(adjacency_matrix_new)
                        )

                    if h_new > 0.25 * h:
                        rho *= 10
                    else:
                        break

                # Step 2.(b): Update alpha
                adjacency_matrix_est, h = adjacency_matrix_new, h_new
                alpha += rho * h

                # Step 2.(c): Break away if h converges to 0
                if h <= h_tol or rho >= rho_max:
                    break

        else:
            adjacency_matrix = torch.nn.Parameter(
                self.backend.zeros(2 * n * n, requires_grad=True)
            )
            h = torch.inf
            lbfgs = LBFGS(
                [adjacency_matrix],
                max_iter=5,
                line_search_fn="strong_wolfe",
            )

            def closure():
                lbfgs.zero_grad()
                loss = torch.abs(
                    self._objective_function(adjacency_matrix, *tuple(self.args))
                )
                print("Loss - ", loss)
                loss.backward(retain_graph=True)
                return loss

            for i in iteration:
                # Step 2.(a): Solve for new values of W and h
                if self.args[1] >= rho_max:
                    break
                print(f"\nNew Iteration no - {i+1}\n")
                lbfgs.step(closure)
                h_new, _ = self._constraint_grad(self._adj(adjacency_matrix))
                print(f"\nh, h_new --> {h, h_new}\n")

                if h == torch.inf or h_new > 0.25 * h:
                    self.args[1] *= 10
                else:
                    print(f"Breaking loop - h, h_new --> {h, h_new}")
                    break

                # Step 2.(b): Update alpha
                h = h_new
                self.args[0] += self.args[1] * h  # alpha += rho*h

                # Step 2.(c): Break away if h converges to 0
                if h <= h_tol or self.args[1] >= rho_max:
                    print(f"\nBreaking outer loop - h, rho -->  {h, self.args[1]}\n")
                    break

            adjacency_matrix_est = adjacency_matrix

        # Step 3: Convert doubled matrix to normal n*n and apply threshold to final weighted matrix
        adjacency_matrix_est = self._adj(adjacency_matrix_est)
        adjacency_matrix_est[self.backend.abs(adjacency_matrix_est) < w_threshold] = 0

        edges = []
        for i in range(n):
            for j in range(n):
                if adjacency_matrix_est[i][j] != 0:
                    edges.append((nodes[i], nodes[j]))

        dag = DAG(ebunch=edges)
        dag.add_nodes_from(nodes)
        return dag
