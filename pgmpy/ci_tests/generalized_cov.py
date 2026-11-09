import numpy as np
import pandas as pd

from pgmpy.utils import preprocess_data
from pgmpy.utils._random import _rename_param_to_rng, _rename_to_rng

from ._base import BaseCITest, _CITestResult, _ResidualMixin


class GeneralizedCov(_ResidualMixin, BaseCITest):
    r"""
    Residual cross-covariance determinant CI test with permutation-based p-values [1].

    This test first residualizes :math:`X` and :math:`Y` with respect to :math:`[1, Z]` using
    an estimator (see :class:`~pgmpy.ci_tests.PillaiTrace` for residualization details). Let
    :math:`R_X \in \mathbb{R}^{n \times p}` and :math:`R_Y \in \mathbb{R}^{n \times q}` be the
    residual matrices and let the residual cross-covariance matrix be

    .. math::
        \widehat{\Sigma}_{XY} = \frac{1}{n - 1} R_X^\top R_Y,

    after column-centering the residuals. When :math:`p = q`, the test statistic is the
    determinant magnitude of :math:`\widehat{\Sigma}_{XY}`:

    .. math::
        T = \left| \det(\widehat{\Sigma}_{XY}) \right|.

    When :math:`p \neq q`, :math:`\widehat{\Sigma}_{XY}` is rectangular and its determinant is
    undefined. In that case, the implementation uses the product of its singular values,
    which reduces to :math:`|\det(\widehat{\Sigma}_{XY})|` in the square case.

    :math:`T` equals zero when :math:`X \perp\!\!\!\perp Y \mid Z` and is positive under
    dependence. Because no closed-form null distribution
    is available, the p-value is computed by a permutation test: rows of :math:`R_X` are
    shuffled repeatedly and :math:`T` is recomputed; the p-value is the proportion of
    permuted statistics that meet or exceed the observed value (with a +1 continuity
    correction following Phipson & Smyth [2]).

    The effect size is the root mean square of the canonical correlations
    :math:`\rho_1, \ldots, \rho_s` between :math:`R_X` and :math:`R_Y`, where :math:`s = \min(p, q)`:

    .. math::
        \sqrt{\frac{1}{s} \sum_{i=1}^{s} \rho_i^2}.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    estimator : estimator instance, optional
        Any sklearn-compatible estimator with ``fit``, ``predict``, and ``predict_proba``
        (if testing discrete variables) methods. If ``None`` (default), uses
        ``RandomForestClassifier`` for categorical targets and ``RandomForestRegressor``
        for continuous targets.

    n_permutations : int, default 1000
        Number of permutations used to estimate the null distribution of :math:`T`.

    rng : int, numpy.random.Generator or None, optional
        Source of randomness, passed to ``numpy.random.default_rng``. If None, fresh entropy is drawn from the operating
        system, so results are not reproducible. When tests run in parallel, only an ``int`` gives reproducible results.

        .. versionchanged:: 1.2.0
            Replaces ``random_state``, which is deprecated and will be removed in v2.0.

    Attributes
    ----------
    statistic_ : float
        The observed determinant-style cross-covariance statistic :math:`T`. Set after
        calling the test.
    p_value_ : float
        Permutation-based p-value. Set after calling the test.
    effect_size_ : float
        The root mean square canonical correlation between the residuals. Set after calling the test.
    estimator_x_ : sklearn-compatible estimator
        The fitted estimator used for predicting X.
    estimator_y_ : sklearn-compatible estimator
        The fitted estimator used for predicting Y.

    References
    ----------
    - :footcite:t:`ankan_textor_2023`
    - :footcite:t:`phipson_smyth_2010`
    """

    _tags = {
        "name": "generalized_cov",
        "data_types": ["discrete", "continuous", "mixed"],
        "default_for": None,
        "requires_data": True,
        "capability:conditioning": True,
    }

    @_rename_to_rng("random_state")
    def __init__(
        self,
        data: pd.DataFrame,
        estimator=None,
        n_permutations: int = 1000,
        rng: int | np.random.Generator | None = None,
        use_cache: bool = True,
    ):
        self.data, self.dtypes = preprocess_data(data)
        self.estimator = estimator
        self.n_permutations = n_permutations
        self.rng = rng
        super().__init__(use_cache=use_cache)

    def set_params(self, **params):
        """Set the parameters of this estimator; the deprecated ``random_state`` is accepted as an alias of ``rng``."""
        _rename_param_to_rng(params, "random_state", type(self).__qualname__)
        return super().set_params(**params)

    @staticmethod
    def _cross_covariance_statistic(res_x: pd.DataFrame, res_y: pd.DataFrame) -> float:
        """
        Compute a determinant-style statistic from the residual cross-covariance matrix.

        For square residual cross-covariance matrices this equals the determinant
        magnitude. For rectangular matrices it uses the product of singular values,
        which is the natural extension of determinant magnitude.
        """
        x = res_x.to_numpy(dtype=float)
        y = res_y.to_numpy(dtype=float)

        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)
        cross_cov = (x.T @ y) / (x.shape[0] - 1)

        singular_values = np.linalg.svd(cross_cov, compute_uv=False)
        return float(np.prod(singular_values))

    def _compute_result(self, X: str, Y: str, Z: list):
        """
        Compute the determinant-style cross-covariance statistic, its permutation p-value and the effect size.

        Parameters
        ----------
        X : str
            The first variable for testing X _|_ Y | Z.
        Y : str
            The second variable for testing X _|_ Y | Z.
        Z : list
            Conditioning variables.

        Returns
        -------
        statistic : float
            The determinant-style residual cross-covariance statistic.
        p_value : float
            Permutation-based p-value.
        effect_size : float
            The root mean square of the canonical correlations between the residuals.
        """
        # Step 1: Compute residuals of X and Y given Z.
        res_x, self.estimator_x_ = self.get_residuals(X, Z)
        res_y, self.estimator_y_ = self.get_residuals(Y, Z)

        if isinstance(res_x, pd.Series):
            res_x = res_x.to_frame()
        if isinstance(res_y, pd.Series):
            res_y = res_y.to_frame()

        # Step 2: Compute the determinant-style statistic from the residual cross-covariance.
        statistic = self._cross_covariance_statistic(res_x, res_y)

        # Step 3: Permutation test to obtain the null distribution of T.
        rng = np.random.default_rng(self.rng)
        perm_stats = np.empty(self.n_permutations)
        for t in range(self.n_permutations):
            idx = rng.permutation(res_x.shape[0])
            res_x_perm = res_x.iloc[idx].reset_index(drop=True)
            perm_stats[t] = self._cross_covariance_statistic(res_x_perm, res_y)

        # Phipson-Smyth +1 correction so that p-value is never exactly zero.
        p_value = (float(np.sum(perm_stats >= statistic)) + 1.0) / (self.n_permutations + 1.0)

        # Step 4: Compute the effect size, the root mean square of the canonical correlations between the residuals.
        bases = []
        for residuals in (res_x, res_y):
            values = residuals.to_numpy(dtype=float)
            left_vectors, singular_values, _ = np.linalg.svd(values - values.mean(axis=0), full_matrices=False)
            tolerance = singular_values.max(initial=0.0) * max(values.shape) * np.finfo(float).eps
            bases.append(left_vectors[:, singular_values > tolerance])
        canonical_correlations = np.linalg.svd(bases[0].T @ bases[1], compute_uv=False)
        effect_size = float(np.sqrt(np.sum(canonical_correlations**2) / min(res_x.shape[1], res_y.shape[1])))

        return _CITestResult(statistic=statistic, p_value=p_value, effect_size=effect_size)
