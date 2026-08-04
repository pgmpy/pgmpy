from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import clone

from pgmpy.structure_score._base import BaseStructureScore


class GeneralizedStructureScore(BaseStructureScore):
    r"""
    A composable continuous structure score that decouples the regression model,
    residual noise distribution, and penalisation strategy.

    The local score for a node :math:`X_i` given parents :math:`\Pi_i` is:

    .. math::
        \text{score}(X_i, \Pi_i) = \log p(\hat{\varepsilon}_i \mid \hat{\theta}) -
        \frac{1}{2} \cdot \text{penalty}(k, n)

    where :math:`\hat{\varepsilon}_i = X_i - \hat{f}(\Pi_i)` are the residuals from
    the fitted estimator, :math:`\hat{\theta}` are the fitted noise distribution
    parameters, :math:`k` is the total parameter count, and :math:`n` is the sample size.

    This single class reproduces several causal discovery scoring methods depending
    on the components passed in:

    +--------------------+-------------------------------+------------------+---------+
    | Method             | estimator                     | noise_dist       | penalty |
    +====================+===============================+==================+=========+
    | CAM                | ``LinearGAM()`` or spline     | ``stats.norm``   | ``None``|
    |                    | pipeline                      |                  |         |
    +--------------------+-------------------------------+------------------+---------+
    | TOPIC (no sig.)    | ``LinearGAM()``               | ``stats.norm``   | ``"aic"``|
    +--------------------+-------------------------------+------------------+---------+
    | LiNGAM-flavoured   | ``LinearRegression()``        | ``stats.laplace``| ``"bic"``|
    +--------------------+-------------------------------+------------------+---------+

    Parameters
    ----------
    data : pd.DataFrame
        Continuous observational data. Each column is a variable.
    estimator : sklearn-style estimator
        Any object implementing ``.fit(X, y)`` and ``.predict(X)``. Examples:
        ``LinearRegression()``, ``LinearGAM()`` (pygam), or an sklearn
        ``Pipeline``. Must be cloneable via ``sklearn.base.clone``.

        .. note::
            Non-sklearn estimators (e.g. pygam, statsmodels) work as long as they
            follow the ``fit`` / ``predict`` convention. If ``clone()`` fails, wrap
            the estimator in a thin sklearn-compatible wrapper.

    noise_dist : scipy.stats continuous distribution, optional
        Must support ``.fit(data)`` and ``.logpdf(data, *params)``.
        Default is ``scipy.stats.norm`` (Gaussian).
    penalty : {None, "aic", "bic"} or callable, optional
        Penalisation applied as ``score = log_L - penalty(k, n) / 2``.

        - ``None``      : no penalty (pure log-likelihood).
        - ``"aic"``     : :math:`k` (Akaike).
        - ``"bic"``     : :math:`k \log n` (Schwarz). **Default.**
        - ``callable``  : any function ``f(k: int, n: int) -> float``.

    n_params : callable or None, optional
        Override for parameter count extraction from the fitted estimator.
        Signature: ``n_params(estimator, X) -> int``. When ``None``,
        :meth:`_n_params` duck-types over statsmodels, pygam, and sklearn
        conventions automatically.

    Examples
    --------
    CAM-style scoring (nonlinear, Gaussian, no penalty):

    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.preprocessing import SplineTransformer
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.pipeline import Pipeline
    >>> from pgmpy.structure_score import GeneralizedStructureScore
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame({"x": rng.standard_normal(200), "y": rng.standard_normal(200)})
    >>> spline_lr = Pipeline([("spline", SplineTransformer()), ("lr", LinearRegression())])
    >>> score = GeneralizedStructureScore(data, estimator=spline_lr, penalty=None)
    >>> isinstance(score.local_score("y", ("x",)), float)
    True

    LiNGAM-flavoured scoring (linear, Laplace, BIC):

    >>> from scipy import stats
    >>> from sklearn.linear_model import LinearRegression
    >>> score = GeneralizedStructureScore(data, LinearRegression(),
    ...                                   noise_dist=stats.laplace, penalty="bic")
    >>> isinstance(score.local_score("y", ("x",)), float)
    True

    References
    ----------
    .. [1] Buhlmann, Peters, Ernest. *CAM: Causal Additive Models.*
           Annals of Statistics, 2014. https://arxiv.org/abs/1310.1533
    .. [2] Rolland et al. *SCORE.* ICML 2022. https://arxiv.org/abs/2203.04413
    .. [3] Xu, Mameche, Vreeken. *TOPIC.* AISTATS 2025.
    .. [4] Schultheiss & Buhlmann. *Pitfalls of Gaussian likelihood scoring.*
           2022. https://arxiv.org/abs/2210.11104
    """

    _tags = {
        "name": "flexible",
        "supported_datatype": "continuous",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(
        self,
        data,
        estimator=None,
        noise_dist=stats.norm,
        penalty="bic",
        n_params=None,
        state_names=None,
    ):
        if estimator is None:
            from sklearn.linear_model import LinearRegression

            estimator = LinearRegression()
        self.estimator = estimator
        self.noise_dist = noise_dist
        self.penalty = penalty
        self.n_params_fn = n_params
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """
        Compute the local structure score for ``variable`` given ``parents``.

        Parameters
        ----------
        variable : str
            Name of the target variable (column in ``self.data``).
        parents : tuple of str
            Names of parent variables. Empty tuple scores the marginal.

        Returns
        -------
        float
            Log-likelihood minus half the penalty term. Higher is better.

        Notes
        -----
        When ``parents`` is empty the score is the log-likelihood of ``variable``
        under the marginal fit of ``noise_dist`` — the correct no-parent baseline
        (log p(y) under the noise model, not conditional on any regressor).

        The score formula is::

            score = log L(residuals | noise_dist) - penalty(k, n) / 2

        where ``k = n_params(estimator) + n_params(noise_dist)`` and ``n`` is
        the number of observations.
        """
        data = self.data
        y = data[variable].to_numpy()
        n = len(y)

        if len(parents) == 0:
            # No parents: score the marginal distribution of y directly.
            # This is log p(y) under noise_dist — the correct baseline.
            params = self.noise_dist.fit(y)
            log_L = float(self.noise_dist.logpdf(y, *params).sum())
            k = len(params)
            return log_L - self._penalty_value(k, n) / 2

        X = data[list(parents)].to_numpy()
        est = clone(self.estimator).fit(X, y)
        residuals = y - est.predict(X)
        params = self.noise_dist.fit(residuals)
        log_L = float(self.noise_dist.logpdf(residuals, *params).sum())
        k = self._n_params(est, X) + len(params)
        return log_L - self._penalty_value(k, n) / 2

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def _n_params(self, est, X: np.ndarray) -> int:
        """
        Extract the number of free parameters from a fitted estimator.

        Duck-types over three common conventions in order:

        1. **statsmodels** — uses ``est.df_model`` (excludes intercept; we add 1
           to account for the intercept that statsmodels fits by default).
        2. **pygam** — uses ``est.statistics_["edof"]`` (effective degrees of
           freedom under penalised splines; the correct count for CAM).
        3. **sklearn-style** — counts ``est.coef_`` entries plus 1 if an
           intercept (``est.intercept_``) was fitted.

        Parameters
        ----------
        est : fitted estimator
            The estimator after calling ``.fit(X, y)``.
        X : np.ndarray
            The design matrix used for fitting (used for shape info if needed).

        Returns
        -------
        int
            Number of free parameters in the fitted estimator.

        Raises
        ------
        AttributeError
            If none of the above conventions apply and no ``n_params`` callable
            was supplied at construction time.
        """
        if self.n_params_fn is not None:
            return int(self.n_params_fn(est, X))

        # sklearn Pipeline: delegate to the final step
        if hasattr(est, "steps"):
            return self._n_params(est[-1], X)

        # statsmodels convention: df_model excludes intercept
        if hasattr(est, "df_model"):
            return int(est.df_model) + 1  # +1 for intercept

        # pygam convention: effective degrees of freedom
        if hasattr(est, "statistics_") and "edof" in est.statistics_:
            return int(est.statistics_["edof"])

        # sklearn convention: coef_ array + optional intercept
        if hasattr(est, "coef_"):
            n = int(np.asarray(est.coef_).size)
            n += int(getattr(est, "intercept_", None) is not None)
            return n

        raise AttributeError(
            "Cannot determine parameter count from the fitted estimator. "
            "Pass `n_params=callable(est, X) -> int` to FlexibleStructureScore. "
            f"Estimator type: {type(est).__name__}."
        )

    # ------------------------------------------------------------------
    # Penalty dispatch
    # ------------------------------------------------------------------

    def _penalty_value(self, k: int, n: int) -> float:
        """
        Compute the raw penalty term (before the /2 division).

        Parameters
        ----------
        k : int
            Total parameter count (estimator + noise distribution).
        n : int
            Number of observations.

        Returns
        -------
        float
            The penalty value. The caller divides by 2 before subtracting.

        Raises
        ------
        ValueError
            If ``self.penalty`` is not one of the supported options.
        """
        if self.penalty is None:
            return 0.0
        if self.penalty == "aic":
            return float(k)
        if self.penalty == "bic":
            return float(k) * np.log(n)
        if callable(self.penalty):
            return float(self.penalty(k, n))
        raise ValueError(f"Unknown penalty: {self.penalty!r}. Use None, 'aic', 'bic', or a callable(k, n) -> float.")
