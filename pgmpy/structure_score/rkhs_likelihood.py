import numpy as np
from sklearn.gaussian_process.kernels import RBF, Kernel

from pgmpy.structure_score._base import BaseStructureScore


class RKHSLikelihood(BaseStructureScore):
    _tags = {
        "name": "ll-rkhs",
        "supported_datatype": "continuous",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None, kernel=None, alpha=1.0, max_cache_size=10000):
        super().__init__(data=data, state_names=state_names, max_cache_size=max_cache_size)
        self._np_data = self.data.to_numpy()
        self._col_index = {col: i for i, col in enumerate(self.data.columns)}
        if kernel is None:
            kernel = RBF()
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be an instance of sklearn.gaussian_process.kernels.Kernel")
        self.kernel = kernel
        self.alpha = alpha

    def _kernel_matrix(self, X):
        K = self.kernel(X)
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def _log_likelihood(self, K_x, K_z):
        n = K_x.shape[0]
        ridge = K_z + n * self.alpha * np.eye(n)

        F_hat = K_x @ np.linalg.solve(ridge, K_z)

        residuals = K_x - F_hat
        residual_cov = residuals @ residuals.T / n

        _, logdet = np.linalg.slogdet(residual_cov)

        return -(n**2 / 2.0) * np.log(2 * np.pi) - (n / 2.0) * logdet - (n / 2.0)

    def _local_score(self, variable, parents):
        y = self._np_data[:, self._col_index[variable]].reshape(-1, 1)
        K_x = self._kernel_matrix(y)
        if parents:
            X = self._np_data[:, [self._col_index[p] for p in parents]]
            K_z = self._kernel_matrix(X)
        else:
            K_z = np.zeros_like(K_x)
        return self._log_likelihood(K_x, K_z)
