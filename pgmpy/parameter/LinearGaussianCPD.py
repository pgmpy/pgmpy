import numpy as np
from sklearn.linear_model import LinearRegression
from skpro.distributions.normal import Normal

from pgmpy.parameter._base import BaseParameter


class LinearGaussianCPD(BaseParameter):
    """
    Estimates a linear Gaussian conditional probability distribution.

    When fit without a target, this estimator learns the marginal Gaussian
    distribution of a root variable. When fit with a target, it learns a
    conditional Gaussian distribution whose mean is a linear function of the
    evidence variables.

    Parameters
    ----------
    estimator: str, default="mle"
        Estimation method used for the variance. If `"mle"`, maximum likelihood
        estimation is used. If `"unbias"`, an unbiased variance estimate is
        used.

    evidences: list, optional
        Ordered list of evidence variable names. If specified, regression
        coefficients in `beta_` are stored in this order. If unspecified, the
        evidence order is inferred from the columns of `X`.

    Attributes
    ----------
    beta_ : numpy.ndarray or list
        Learned parameters of the linear Gaussian distribution. For a root
        variable, this contains only the learned mean. For a conditional
        distribution, the first value is the intercept and the remaining values
        are regression coefficients ordered by `evidences_`. Populated by `fit`.

    std_ : float
        Learned standard deviation of the root variable or of the regression
        residuals. Populated by `fit`.

    evidences_ : list or None
        Ordered list of evidence variables used during fitting and prediction.
        Set to `None` for root-variable distributions. Populated by `fit`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.parameter.LinearGaussianCPD import LinearGaussianCPD
    >>> rng = np.random.default_rng(seed=42)
    >>> n = 1000
    >>> A = rng.normal(size=n)
    >>> B = rng.normal(size=n)
    >>> C = rng.normal(size=n)
    >>> mean_y = 1 + 2 * A + 5 * B + 7 * C
    >>> X = pd.DataFrame(
    ...     {
    ...         "A": A,
    ...         "B": B,
    ...         "C": C,
    ...     }
    ... )
    >>> y = pd.DataFrame({"y": rng.normal(loc=mean_y, scale=3)})
    >>> cpd = LinearGaussianCPD()
    >>> cpd.fit(X, y)
    LinearGaussianCPD()
    >>> dist = cpd.predict_proba(X[:5])

    """

    _tags = {
        "object_type": {"continuous"},
        "fit_mode": {"supervise", "unsupervise"},
        "python_dependencies": set(),
        "local:plug_in": {"mle", "unbias"},
        "global:plug_in": set(),
        "local:full_bayesian": set(),
        "global:full_bayesian": set(),
    }

    def __init__(self, estimator="mle", evidences=None):
        self.estimator = estimator
        self.evidences = evidences

    def _fit(self, X, y=None, sample_weight=None):
        if y is None:
            x_arr = X.to_numpy(dtype=float).reshape(-1)
            y_arr = None
            self.evidences_ = None
            if self.estimator == "mle":
                ddof = 0
            elif self.estimator == "unbias":
                ddof = 1
        else:
            y_arr = y.to_numpy(dtype=float).reshape(-1)
            if self.evidences is not None:
                self.evidences_ = list(self.evidences)
            else:
                self.evidences_ = list(X.columns)
            if self.estimator == "mle":
                ddof = 0
            elif self.estimator == "unbias":
                ddof = 1 + len(self.evidences_)

        if y_arr is None:
            # Unsupervised Learning
            w_mean = np.sum(sample_weight * x_arr) / np.sum(sample_weight)
            w_var = np.sum(sample_weight * (x_arr - w_mean) ** 2) / (np.sum(sample_weight) - ddof)

            self.beta_ = [w_mean]
            self.std_ = np.sqrt(w_var)
        else:
            # Supervised Learning
            X_fit = X.loc[:, self.evidences_]

            lm = LinearRegression().fit(X_fit, y_arr, sample_weight)
            residuals = y_arr - lm.predict(X_fit).reshape(-1)
            self.beta_ = np.concatenate(
                [
                    np.ravel(lm.intercept_),
                    np.ravel(lm.coef_),
                ]
            )
            self.std_ = np.sqrt(np.sum(sample_weight * residuals**2) / (np.sum(sample_weight) - ddof))

        return self

    def _predict_proba(self, X):
        intercept = self.beta_[0]
        coef = self.beta_[1:]
        mu = intercept + X.loc[:, self.evidences_].to_numpy() @ coef
        mu = np.asarray(mu, dtype=float).reshape(-1, 1)
        return Normal(mu=mu, sigma=self.std_, index=X.index)

    def _sample(self, X, n_samples):
        if self.fit_mode_ == "unsupervise":
            # Unsupervised Learning
            mu = np.full(n_samples, self.beta_[0], dtype=float)
            mu = np.asarray(mu, dtype=float).reshape(-1, 1)
            dist = Normal(mu=mu, sigma=self.std_)
            samples = dist.sample().to_numpy()
            return samples, np.array(dist.log_pdf(samples))

        elif self.fit_mode_ == "supervise":
            dist = self._predict_proba(X)
            samples = dist.sample().to_numpy()
            return samples, np.array(dist.log_pdf(samples))

    def set_fitted_params(self, beta, std, evidences, is_fitted):
        self.beta_ = beta
        self.std_ = std
        self.evidences_ = evidences
        self._is_fitted = is_fitted
        return self
