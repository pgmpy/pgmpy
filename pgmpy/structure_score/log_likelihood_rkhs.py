from collections.abc import Hashable

import numpy as np

from pgmpy.structure_score._base import BaseStructureScore


class RKHSLikelihood(BaseStructureScore):
    r"""
    Log-likelihood structure score for continuous bayesian networks using RKHS

    This score evaluates a continuous bayesian network structure by modeling conditional
    and unconditional distributions in a Reproducing kernel Hilbert Space (RKHS).

    .. math::
        \operatorname{max rkhs log likelihood}(X, Z) =
        \begin{cases}
        -\frac{n}{2}\log(2\pi) - \frac{1}{2}\log|\tilde{K}_X + n\lambda I| - \frac{n}{2}, & \text{if } Z = \emptyset \\
        \begin{aligned}
        & -\frac{n^2}{2}\log(2\pi) - \frac{n}{2}\log \left| n\lambda^2 \tilde{K}_X \right. \\
        & \left. (\tilde{K}_{Z} + n\lambda I)^{-2} \tilde{K}_X \right| - \frac{n}{2},
        \end{aligned} & \text{if } Z \neq \emptyset
        \end{cases}

    where :math:`n` is the sample size, :math:`\lambda` is the regularization parameter,
    :math:`\tilde{K}_X` is the centralized kernel matrix of `X`, defined as `\tilde{K}_X` = `HK_XH`,
    :math:`\tilde{K}_Z` is the centralized kernel matrix of parent configuration Z.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.
    state_names : dict, optional
        Dictionary mapping each variable to its allowed states.
    kernel : str, default="rbf"
        The kernel type to use (default: "rbf").
    gamma : float, optional
        kernel width parameter, If None, it is set to `1 / X.shape[1]`.
    reg_param : float, default=1.0
        Ridge regularization parameter `\lambda`
    max_cache_size : int or None, default=10000
        Maximum number of local scores to cache. If None, the cache is unlimited.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.structure_score import RKHSLikelihood
    >>> df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    >>> score = RKHSLikelihood(df)
    >>> score.local_score("B", ("A",))
    np.float64(-3.1068063509697152)

    References
    ----------
    - :cite:p:`huang_2018`
    Raises
    ------
    ValueError
        If data contains missing values, reg_param is zero or negative,
        kernel is not "rbf", or max_cache_size is not None and is
        zero or negative.

    """

    _tags = {
        "name": "rkhs-likelihood",
        "data_types": ["continuous"],
        "default_for": None,
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": True,
    }

    def __init__(
        self,
        data,
        state_names=None,
        kernel="rbf",
        gamma=None,
        reg_param=1.0,
        max_cache_size=10000,
    ):
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)

        if data.isnull().any().any():
            raise ValueError("Data contains missing values. RKHSLikelihood does not support missing data.")

        if reg_param <= 0:
            raise ValueError("reg_param must be a positive number.")

        if kernel not in ["rbf"]:
            raise ValueError(f"Kernel '{kernel}' is not supported. Only 'rbf' is supported.")

        self.kernel = kernel
        self.gamma = gamma
        self.reg_param = reg_param

    def _kernel_matrix(self, X, Y=None):
        if Y is None:
            Y = X

        gamma = self.gamma
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True).T
        dist_sq = X_norm + Y_norm - (2 * np.dot(X, Y.T))

        dist_sq = np.maximum(dist_sq, 0.0)

        return np.exp(-gamma * dist_sq)

    def _centralized_kernel_matrix(self, X, Y=None):
        is_symmetric = Y is None

        n_x = X.shape[0]

        K = self._kernel_matrix(X, Y)
        H_x = np.eye(n_x) - np.ones((n_x, n_x)) / n_x

        if is_symmetric:
            return H_x @ K @ H_x
        else:
            n_y = Y.shape[0]
            H_y = np.eye(n_y) - np.ones((n_y, n_y)) / n_y
            return H_x @ K @ H_y

    def _residual_covariance(self, K_Z):
        n = K_Z.shape[0]
        reg_K_Z = K_Z + n * self.reg_param * np.eye(n)
        return np.linalg.inv(reg_K_Z)

    def _log_likelihood(self, variable: Hashable, parents: tuple[Hashable, ...]) -> float:
        X = self.data[[variable]].values
        n = X.shape[0]
        K_X = self._centralized_kernel_matrix(X)
        I = np.eye(n)

        # Unconditional case (no parents)
        if not parents:
            cov = K_X + n * self.reg_param * I
            _, log_det = np.linalg.slogdet(cov)
            score = (-0.5 * n * np.log(2 * np.pi)) - (0.5 * log_det) - 0.5 * n
            return np.float64(score)

        # Conditional case (with parents)
        Z = self.data[list(parents)].values
        K_Z = self._centralized_kernel_matrix(Z)
        inv_reg_K_Z = self._residual_covariance(K_Z)

        term = n * (self.reg_param**2) * K_X @ (inv_reg_K_Z @ inv_reg_K_Z) @ K_X

        eigenvals = np.linalg.eigvalsh(term)
        valid_eigenvals = eigenvals[eigenvals > 1e-10]
        log_det = np.sum(np.log(valid_eigenvals))

        score = -0.5 * (n**2) * np.log(2 * np.pi) - 0.5 * n * log_det - 0.5 * n
        return np.float64(score)

    def _local_score(self, variable: Hashable, parents: tuple[Hashable, ...]) -> float:
        return self._log_likelihood(variable, parents)
