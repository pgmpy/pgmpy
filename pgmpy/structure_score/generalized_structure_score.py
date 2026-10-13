from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, clone

from pgmpy.structure_score._base import BaseStructureScore


class GeneralizedStructureScore(BaseStructureScore):
    r"""
    Structure score for continuous data with configurable regression, noise, and penalty.

    Each local score fits a regressor to the parent variables, then evaluates the
    residuals under the noise distribution:

    .. math::
        \operatorname{score} = \sum_{j=1}^{n} \log p(\hat{\varepsilon}_j \mid \hat{\theta})
        - \frac{1}{2} \operatorname{penalty}(k, n),

    where :math:`\hat{\varepsilon}_j` is a fitted residual, :math:`\hat{\theta}` contains the
    noise parameters, :math:`k` is the total parameter count, and :math:`n` is the sample size.
    Higher scores are preferred. For nodes without parents, the noise distribution is fitted
    directly to the observed values.

    Configurations for common scores:

    +-------------------------+------------------------------------+-------------------+------------------+
    | Scoring method          | ``estimator``                      | ``noise_dist``    | ``penalty``      |
    +=========================+====================================+===================+==================+
    | Gaussian log-likelihood | ``LinearRegression()``             | ``stats.norm``    | ``None``         |
    +-------------------------+------------------------------------+-------------------+------------------+
    | Gaussian AIC            | ``LinearRegression()``             | ``stats.norm``    | ``"aic"``        |
    +-------------------------+------------------------------------+-------------------+------------------+
    | Gaussian BIC            | ``LinearRegression()``             | ``stats.norm``    | ``"bic"``        |
    +-------------------------+------------------------------------+-------------------+------------------+
    | Gaussian AICc           | ``LinearRegression()``             | ``stats.norm``    | ``aicc_penalty`` |
    +-------------------------+------------------------------------+-------------------+------------------+
    | Gaussian HQIC           | ``LinearRegression()``             | ``stats.norm``    | ``hqic_penalty`` |
    +-------------------------+------------------------------------+-------------------+------------------+
    | CAM-style               | ``LinearGAM()`` or spline pipeline | ``stats.norm``    | ``None``         |
    +-------------------------+------------------------------------+-------------------+------------------+
    | TOPIC-style             | ``LinearGAM()``                    | ``stats.norm``    | ``"aic"``        |
    +-------------------------+------------------------------------+-------------------+------------------+
    | LiNGAM-flavoured        | ``LinearRegression()``             | ``stats.laplace`` | ``"bic"``        |
    +-------------------------+------------------------------------+-------------------+------------------+

    The Gaussian rows assume a full-rank linear regression with an intercept and
    count the noise scale as a parameter. Information criteria are returned as
    ``-IC / 2``; AICc is applied separately to each node. The callable penalties
    are defined in the examples below.

    ``LinearGAM`` is from pyGAM. The CAM, TOPIC, and LiNGAM rows are related
    configurations; matching those methods requires matching their regression
    settings, noise estimation, and parameter counts.

    Parameters
    ----------
    data : pandas.DataFrame
        Continuous observations, with one variable per column.
    estimator : regressor or None, default=None
        Unfitted regressor implementing ``fit(X, y)`` and ``predict(X)``.
        ``None`` uses ``sklearn.linear_model.LinearRegression``. The regressor
        is cloned or deep-copied before each fit. Estimators inside an sklearn
        pipeline must support cloning without losing their configuration.
    noise_dist : object, default=scipy.stats.norm
        Distribution implementing ``fit(data)`` and ``logpdf(data, *params)``.
        Each parameter returned by ``fit`` contributes to the parameter count.
        For scipy.stats distributions, an estimator intercept and the noise
        location are counted as one shared parameter.
    penalty : {None, "aic", "bic"} or callable, default="bic"
        Raw penalty before division by two: ``None`` gives zero, ``"aic"`` gives
        :math:`2k`, and ``"bic"`` gives :math:`k \log n`. A callable must accept
        ``(k, n)`` and return the raw penalty.
    n_params : callable or None, default=None
        ``n_params(estimator, X) -> int`` returns the fitted regressor's parameter
        count, including its intercept and excluding noise parameters. ``X`` is
        the array of parent observations. If ``None``, the count is inferred from
        ``df_model``, ``statistics_["edof"]``, or ``coef_``; other estimators require
        a callback.
    state_names : dict or None, default=None
        Mapping of variables to allowed states, usually omitted for continuous data.

    Examples
    --------
    Define the AICc and Hannan--Quinn (HQIC) penalties used in the table:

    >>> import numpy as np
    >>> def aicc_penalty(k, n):
    ...     return 2 * k * n / (n - k - 1) if n > k + 1 else np.inf
    >>> def hqic_penalty(k, n):
    ...     return 2 * k * np.log(np.log(n))

    Score a spline regression with Gaussian noise and no penalty:

    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import SplineTransformer
    >>> from pgmpy.structure_score import GeneralizedStructureScore
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(200)
    >>> data = pd.DataFrame({"x": x, "y": x**2 + rng.standard_normal(200)})
    >>> estimator = make_pipeline(SplineTransformer(), LinearRegression())
    >>> score = GeneralizedStructureScore(data, estimator=estimator, penalty=None)
    >>> round(score.local_score("y", ("x",)), 3)
    -284.923

    References
    ----------
    - Likelihood foundations: :cite:p:`fisher_1922`.
    - AIC: :cite:p:`akaike_1973,akaike_1974`.
    - BIC: :cite:p:`schwarz_1978`.
    - AICc: :cite:p:`sugiura_1978,hurvich_tsai_1989`.
    - HQIC: :cite:p:`hannan_quinn_1979`.
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
        self.n_params = n_params
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """Compute the penalized log-likelihood for a node and its parents."""
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
        if hasattr(self.estimator, "__sklearn_clone__"):
            est = self.estimator.__sklearn_clone__()
        elif isinstance(self.estimator, BaseEstimator):
            est = clone(self.estimator)
        else:
            est = deepcopy(self.estimator)
        est = est.fit(X, y)
        residuals = y - est.predict(X)
        params = self.noise_dist.fit(residuals)
        log_L = float(self.noise_dist.logpdf(residuals, *params).sum())
        k = self._n_params(est, X) + len(params)
        if isinstance(self.noise_dist, stats.rv_continuous) and self._has_intercept(est):
            k -= 1
        return log_L - self._penalty_value(k, n) / 2

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def _has_intercept(self, est: object) -> bool:
        """Return whether the fitted estimator includes an intercept."""
        if hasattr(est, "steps"):
            return self._has_intercept(est[-1])
        if hasattr(est, "fit_intercept"):
            return bool(est.fit_intercept)
        if hasattr(est, "k_constant"):
            return bool(est.k_constant)
        if hasattr(est, "df_model"):
            return True
        return getattr(est, "intercept_", None) is not None

    def _n_params(self, est, X: np.ndarray) -> int:
        """Return the regression parameter count, excluding noise parameters.

        Use the callback or inspect the estimator, using the final step for pipelines.
        Raise ``AttributeError`` if no count can be determined.
        """
        if self.n_params is not None:
            return int(self.n_params(est, X))

        # sklearn Pipeline: delegate to the final step
        if hasattr(est, "steps"):
            return self._n_params(est[-1], X)

        # statsmodels convention: df_model excludes intercept
        if hasattr(est, "df_model"):
            return int(est.df_model) + int(self._has_intercept(est))

        # pygam convention: effective degrees of freedom
        if hasattr(est, "statistics_") and "edof" in est.statistics_:
            return int(est.statistics_["edof"])

        # sklearn convention: coef_ array + optional intercept
        if hasattr(est, "coef_"):
            n = int(np.asarray(est.coef_).size)
            n += int(self._has_intercept(est))
            return n

        raise AttributeError(
            "Cannot determine parameter count from the fitted estimator. "
            "Pass `n_params=callable(est, X) -> int` to GeneralizedStructureScore. "
            f"Estimator type: {type(est).__name__}."
        )

    # ------------------------------------------------------------------
    # Penalty dispatch
    # ------------------------------------------------------------------

    def _penalty_value(self, k: int, n: int) -> float:
        """Return the penalty for ``k`` parameters and ``n`` observations, before halving."""
        if self.penalty is None:
            return 0.0
        if self.penalty == "aic":
            return 2.0 * k
        if self.penalty == "bic":
            return float(k) * np.log(n)
        if callable(self.penalty):
            return float(self.penalty(k, n))
        raise ValueError(f"Unknown penalty: {self.penalty!r}. Use None, 'aic', 'bic', or a callable(k, n) -> float.")
