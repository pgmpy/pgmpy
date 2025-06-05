import numpy as np
import pandas as pd
from numpy.linalg import pinv


class LinearEstimator:
    """
    Lightweight linear‐regression wrapper that supports
    ▸ OLS  ▸ GLS  ▸ WLS
    with no external statsmodels/scikit-learn dependency.
    """

    def __init__(self, graph=None, estimator_type="linear", **kwargs):
        self._supported = {"linear": "OLS", "OLS": "OLS", "GLS": "GLS", "WLS": "WLS"}

        if estimator_type not in self._supported:
            raise NotImplementedError("Supported types: OLS, GLS, WLS")

        self.estimator_type = self._supported[estimator_type]
        self.params: np.ndarray | None = None
        self.ate: float | None = None
        self.results: dict[str, np.ndarray | float] = {}
        self.feature_names: list[str] = []

    # helpers
    @staticmethod
    def _add_constant(X: np.ndarray) -> np.ndarray:
        """prepend a column of 1 s (intercept)"""
        return np.column_stack([np.ones(X.shape[0]), X])

    # --------------------------- core solvers -------------------------- #
    def _ols(self, X, y):
        X = self._add_constant(X)
        XtX = X.T @ X
        beta = pinv(XtX) @ X.T @ y

        residuals = y - X @ beta
        sigma2 = (residuals.T @ residuals) / (X.shape[0] - X.shape[1])
        var_cov = sigma2 * pinv(XtX)
        se = np.sqrt(np.diag(var_cov))

        return beta, residuals, se, var_cov, sigma2

    def _wls(self, X, y, weights):
        X = self._add_constant(X)
        W = np.diagflat(weights)
        XtWX = X.T @ W @ X
        beta = pinv(XtWX) @ X.T @ W @ y

        residuals = y - X @ beta
        sigma2 = (residuals.T @ W @ residuals) / (X.shape[0] - X.shape[1])
        var_cov = sigma2 * pinv(XtWX)
        se = np.sqrt(np.diag(var_cov))

        return beta, residuals, se, var_cov, sigma2

    def _gls(self, X, y, omega):
        X = self._add_constant(X)
        omega_inv = pinv(omega)
        XtOinvX = X.T @ omega_inv @ X
        beta = pinv(XtOinvX) @ X.T @ omega_inv @ y

        residuals = y - X @ beta
        sigma2 = (residuals.T @ omega_inv @ residuals) / (X.shape[0] - X.shape[1])
        var_cov = sigma2 * pinv(XtOinvX)
        se = np.sqrt(np.diag(var_cov))

        return beta, residuals, se, var_cov, sigma2

    # internal model dispatcher
    def _run(self, X, y, **kwargs):
        if self.estimator_type == "OLS":
            return self._ols(X, y)

        if self.estimator_type == "WLS":
            weights = kwargs.get("weights")
            if weights is None:
                return self._ols(X, y)
            return self._wls(X, y, weights)

        if self.estimator_type == "GLS":
            omega = kwargs.get("sigma")
            if omega is None:
                return self._ols(X, y)
            return self._gls(X, y, omega)

    # public API
    def fit(self, X: str, Y: str, Z: list[str], data: pd.DataFrame, **kwargs):
        """Fit the chosen estimator and store results"""
        self.feature_names = [X] + list(Z)

        X_mat = data[self.feature_names].to_numpy()
        y_vec = data[Y].to_numpy()

        beta, resid, se, var_cov, sigma2 = self._run(X_mat, y_vec, **kwargs)

        self.params = beta
        # ATE is coefficient on treatment X (index 1; index 0 is intercept)
        self.ate = beta[1]
        self.results = {
            "params": beta,
            "residuals": resid,
            "std_errors": se,
            "var_cov": var_cov,
            "sigma_squared": sigma2,
        }
        return self

    def _get_ate(self) -> float:
        """Return Average Treatment Effect (coef on X)"""
        if self.ate is None:
            raise RuntimeError("Model not yet fitted.")
        return self.ate

    # summary
    def summary(self) -> str:
        if not self.results:
            return "Model not yet fitted."

        lines = [
            f"{self.estimator_type} Regression Results",
            "-" * 50,
            f"ATE (coef on '{self.feature_names[0]}'): {self.ate:.4f}",
            "",
            "Coefficients:",
        ]

        names = ["Intercept"] + self.feature_names
        for name, coef, se in zip(names, self.params, self.results["std_errors"]):
            lines.append(f"{name:>15}: {coef:>10.4f} (SE: {se:.4f})")

        lines.append(f"\nResidual variance (σ²): {self.results['sigma_squared']:.4f}")
        return "\n".join(lines)

    # convenience representation
    def __repr__(self):
        if self.params is None:
            return f"<LinearEstimator [{self.estimator_type}] – not yet fitted>"
        return f"<LinearEstimator [{self.estimator_type}] ATE={self.ate:.4f}>"
