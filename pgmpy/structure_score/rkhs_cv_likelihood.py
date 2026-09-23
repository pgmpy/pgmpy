import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold

from pgmpy.structure_score import RKHSLikelihood


class CrossValidatedRKHSLikelihood(RKHSLikelihood):
    _tags = {
        "name": "rkhs-cv-ll",
        "supported_datatype": "continuous",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(
        self,
        data,
        state_names=None,
        kernel="rbf",
        gamma=None,
        alpha=1.0,
        fold=10,
        random_state=42,
        max_cache_size=10000,
    ):
        super().__init__(
            data=data, state_names=state_names, kernel=kernel, gamma=gamma, alpha=alpha, max_cache_size=max_cache_size
        )
        self.fold = fold
        self.random_state = random_state
        if self.fold > len(self.data):
            raise ValueError(f"fold={fold} cannot exceed number of samples={len(self.data)}")

    def _cross_kernel_matrix(self, X_test, X_train):
        K01 = pairwise_kernels(X_test, X_train, metric=self.kernel, gamma=self.gamma)
        K11 = pairwise_kernels(X_train, X_train, metric=self.kernel, gamma=self.gamma)
        return K01 - K01.mean(axis=1, keepdims=True) - K11.mean(axis=0, keepdims=True) + K11.mean()

    def _cv_log_likelihood(self, K_x_train, K_x_test, K_z_train, K_z_test, n_train, n_test):
        """
        CV log-likelihood: fit on training fold, evaluate on test fold (Eq. 8 / Appendix A2).
        """
        # Predict training
        ridge = K_z_train + n_train * self.alpha * np.eye(n_train)
        kernel_regression = np.linalg.solve(ridge, K_z_train)
        F_hat_train = K_x_train @ kernel_regression

        # Training residuals
        residuals = K_x_train - F_hat_train
        residual_cov_train = residuals @ residuals.T / n_train

        _, logdet = np.linalg.slogdet(residual_cov_train)

        # Predict test
        F_hat_test = K_z_test @ np.linalg.solve(ridge, K_x_train)

        # Test residuals
        residual_test = K_x_test - F_hat_test

        trace_matrix = residual_cov_train + self.alpha * np.eye(n_train)
        quad = np.trace(residual_test.T @ np.linalg.solve(trace_matrix, residual_test))

        return -(n_test**2 / 2.0) * np.log(2 * np.pi) - (n_test / 2.0) * logdet - (n_test / 2.0) * quad

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        kf = KFold(n_splits=self.fold, shuffle=True, random_state=self.random_state)

        y = self._np_data[:, self._col_index[variable]].reshape(-1, 1)
        X = self._np_data[:, [self._col_index[p] for p in parents]] if parents else None

        scores = []
        for train_idx, test_idx in kf.split(y):
            n_train, n_test = len(train_idx), len(test_idx)
            y_train, y_test = y[train_idx], y[test_idx]

            K_x_train = self._kernel_matrix(y_train)
            K_x_test = self._cross_kernel_matrix(y_test, y_train)

            if parents:
                X_train, X_test = X[train_idx], X[test_idx]
                K_z_train = self._kernel_matrix(X_train)
                K_z_test = self._cross_kernel_matrix(X_test, X_train)
            else:
                K_z_train = np.zeros((n_train, n_train))
                K_z_test = np.zeros((n_test, n_train))

            scores.append(self._cv_log_likelihood(K_x_train, K_x_test, K_z_train, K_z_test, n_train, n_test))

        return np.mean(scores)
