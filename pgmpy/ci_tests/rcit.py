import numpy as np
import pandas as pd
from scipy import stats

from ._base import _BaseCITest


def _median_bandwidth(x: np.ndarray) -> float:
    """Compute Gaussian kernel bandwidth via the median heuristic."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    if n > 500:
        idx = np.linspace(0, n - 1, 500, dtype=int)
        x = x[idx]
    dists_sq = np.sum((x[:, None] - x[None, :]) ** 2, axis=-1)
    upper = dists_sq[np.triu_indices_from(dists_sq, k=1)]
    if upper.size == 0 or upper.max() == 0:
        return 1.0
    bw = float(np.sqrt(np.median(upper)))
    if not np.isfinite(bw) or bw <= 0.0:
        return 1.0
    return bw


def _normalize(x: np.ndarray) -> np.ndarray:
    """Standardize each column to zero mean and unit variance."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (x - mu) / sigma


def _rff(x: np.ndarray, num_features: int, bandwidth: float, rng: np.random.Generator) -> np.ndarray:
    """
    Compute normalized random Fourier features for a Gaussian kernel.

    Uses the cosine approximation: phi(x) = sqrt(2/r) * cos(omega^T x + b)
    where omega ~ N(0, I/bandwidth^2) and b ~ Uniform(0, 2*pi).
    The output is normalized to zero mean and unit variance per column.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    d = x.shape[1]
    omega = rng.standard_normal((d, num_features)) / bandwidth
    b = rng.uniform(0, 2 * np.pi, num_features)
    phi = np.sqrt(2.0 / num_features) * np.cos(x @ omega + b)
    return _normalize(phi)


def _partial_covariance(f_x, f_y, f_z, reg=1e-10):
    """
    Compute the partial cross-covariance C_{xy.z} = C_{xy} - C_{xz} * inv(C_{zz} + reg*I) * C_{zy}.

    Also returns the residuals res_x = f_x - f_z * inv(C_{zz} + reg*I) * C_{xz}.T
    and res_y = f_y - f_z * inv(C_{zz} + reg*I) * C_{zy} used for the null distribution.
    """
    num_z = f_z.shape[1]

    # Sample covariances (using n-1 denominator via np.cov).
    # np.cov with two arguments returns a (p+q)x(p+q) block matrix.
    fxz = np.column_stack([f_x, f_z])
    fyz = np.column_stack([f_y, f_z])
    num_x = f_x.shape[1]
    num_y = f_y.shape[1]

    cov_xz_block = np.cov(fxz, rowvar=False)
    cov_yz_block = np.cov(fyz, rowvar=False)
    cov_xy_block = np.cov(np.column_stack([f_x, f_y]), rowvar=False)

    C_xz = cov_xz_block[:num_x, num_x:]  # num_x x num_z
    C_zy = cov_yz_block[num_y:, :num_y]  # num_z x num_y
    C_zz = np.cov(f_z, rowvar=False)  # num_z x num_z
    C_xy = cov_xy_block[:num_x, num_x:]  # num_x x num_y

    reg_Czz = C_zz + np.eye(num_z) * reg
    solve_Czy = np.linalg.solve(reg_Czz, C_zy)
    solve_Cxz_t = np.linalg.solve(reg_Czz, C_xz.T)

    # Partial cross-covariance.
    C_xy_z = C_xy - C_xz @ solve_Czy

    # Residuals for the null distribution.
    res_x = f_x - f_z @ solve_Cxz_t  # n x num_x
    res_y = f_y - f_z @ solve_Czy  # n x num_y

    return C_xy_z, res_x, res_y


def _null_eigenvalues(res_x: np.ndarray, res_y: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of the residual cross-product covariance matrix.

    Forms all (num_x * num_y) elementwise products of residual columns and
    returns the positive eigenvalues of their sample covariance, which serve
    as weights in the weighted chi-squared null distribution.
    """
    n = res_x.shape[0]
    num_x, num_y = res_x.shape[1], res_y.shape[1]
    # All pairwise elementwise products: n x (num_x * num_y)
    pairs = np.array([(i, j) for i in range(num_x) for j in range(num_y)])
    cross = res_x[:, pairs[:, 0]] * res_y[:, pairs[:, 1]]
    Cov = (cross.T @ cross) / n
    eigs = np.linalg.eigvalsh(Cov)
    return eigs[eigs > 0]


def _gamma_pvalue(statistic: float, lambdas: np.ndarray) -> float:
    """
    P-value via the Satterthwaite/Gamma approximation for a weighted chi-squared null.

    Under H0, statistic ~_d sum_i lambda_i * chi2(1).
    Approximated by c * chi2(k) where:
      c = sum(lambda_i^2) / sum(lambda_i)
      k = sum(lambda_i)^2 / sum(lambda_i^2)
    """
    lambdas = lambdas[lambdas > 1e-10]
    if len(lambdas) == 0:
        return 1.0
    s1 = float(lambdas.sum())
    s2 = float((lambdas**2).sum())
    c = s2 / s1
    k = s1**2 / s2
    return float(stats.chi2.sf(statistic / c, df=k))


class RCIT(_BaseCITest):
    r"""
    Randomised Conditional Independence Test (RCIT) for continuous data.

    RCIT approximates a kernel-based conditional independence test using Random
    Fourier Features (RFF) for X, Y augmented with Z, and Z. The key property
    of RCIT is that the feature map for Y is computed on the joint :math:`[Y, Z]`
    vector, which increases the sensitivity of the test to the :math:`Y`–:math:`Z`
    relationship. It tests :math:`X \perp Y \mid Z` by computing the partial
    cross-covariance :math:`\hat{C}_{XY \mid Z} = \hat{C}_{XY} - \hat{C}_{XZ}
    \hat{C}_{ZZ}^{-1} \hat{C}_{ZY}` in the random feature space.

    Under the null hypothesis :math:`X \perp Y \mid Z`, the scaled test statistic
    :math:`n \cdot \|\hat{C}_{XY \mid Z}\|_F^2` follows an approximate weighted
    :math:`\chi^2` distribution. The weights are the eigenvalues of the residual
    cross-product covariance matrix, and the p-value is obtained via a
    Satterthwaite/Gamma approximation.

    When Z is empty the test falls back to :class:`Pearsonr`.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.
    num_features_x : int, default=5
        Number of random Fourier features for X.
    num_features_y : int, default=5
        Number of random Fourier features for Y (computed on [Y, Z]).
    num_features_z : int, default=100
        Number of random Fourier features for Z.
    seed : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    statistic_ : float
        The RCIT test statistic :math:`n \cdot \|\hat{C}_{XY \mid Z}\|_F^2`.
        When Z is empty the fallback :class:`Pearsonr` is used and
        ``statistic_`` holds Pearson's r instead.
        Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import RCIT
    >>> rng = np.random.default_rng(seed=42)
    >>> Z = rng.standard_normal(1000)
    >>> X = Z + rng.standard_normal(1000)
    >>> Y = Z + rng.standard_normal(1000)
    >>> data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    >>> test = RCIT(data=data, seed=0)
    >>> test("X", "Y", ["Z"], significance_level=0.05)
    True

    References
    ----------
    .. [1] Strobl, E. V., Zhang, K., & Visweswaran, S. (2019). Approximate
       Kernel-Based Conditional Independence Tests for Fast Non-Parametric
       Causal Discovery. Statistics and Computing, 29(5), 891-912.
       https://arxiv.org/abs/1702.03877
    """

    _tags = {
        "name": "rcit",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        num_features_x: int = 5,
        num_features_y: int = 5,
        num_features_z: int = 100,
        seed: int | None = None,
    ):
        self.data = data
        self.num_features_x = num_features_x
        self.num_features_y = num_features_y
        self.num_features_z = num_features_z
        self.seed = seed
        super().__init__()

    def run_test(self, X: str, Y: str, Z: list):
        """
        Compute the RCIT statistic and p-value.

        Sets ``self.statistic_`` and ``self.p_value_``.
        """
        data = self.data
        n = data.shape[0]
        rng = np.random.default_rng(self.seed)

        x = data[X].to_numpy(dtype=float)
        y = data[Y].to_numpy(dtype=float)

        # Step 1: Fall back to Pearsonr for the unconditional case.
        if len(Z) == 0:
            from .pearsonr import Pearsonr

            stat, pval = Pearsonr(data=data).run_test(X=X, Y=Y, Z=[])
            self.statistic_ = float(stat)
            self.p_value_ = float(pval)
            return self.statistic_, self.p_value_

        z = data[Z].to_numpy(dtype=float)
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        # Step 2: Normalize x, y, z to zero mean and unit variance.
        x = _normalize(x)
        y = _normalize(y)
        z = _normalize(z)

        # Step 3: RCIT augments y with z before computing its feature map.
        # This is the defining difference from RCoT.
        y_aug = np.column_stack([y, z])

        # Step 4: Compute bandwidths via median heuristic on the normalized data.
        bw_x = _median_bandwidth(x)
        bw_y = _median_bandwidth(y_aug)
        bw_z = _median_bandwidth(z)

        # Step 5: Compute normalized random Fourier features.
        f_x = _rff(x, self.num_features_x, bw_x, rng)
        f_y = _rff(y_aug, self.num_features_y, bw_y, rng)
        f_z = _rff(z, self.num_features_z, bw_z, rng)

        # Step 6: Partial cross-covariance and residuals via covariance projection.
        C_xy_z, res_x, res_y = _partial_covariance(f_x, f_y, f_z)

        # Step 7: Test statistic.
        statistic = float(n * np.sum(C_xy_z**2))

        # Step 8: Eigenvalues of residual cross-product covariance for the null.
        lambdas = _null_eigenvalues(res_x, res_y)

        # Step 9: P-value via Gamma approximation.
        p_value = _gamma_pvalue(statistic, lambdas)

        self.statistic_ = statistic
        self.p_value_ = max(0.0, p_value)
        return self.statistic_, self.p_value_


class RCoT(_BaseCITest):
    r"""
    Randomised Conditional Covariance Test (RCoT) for continuous data.

    RCoT is a variant of RCIT. Unlike RCIT, the feature map for Y is computed
    on Y alone (not on [Y, Z]). Both X and Y use small random feature maps while
    Z uses a larger one. RCoT tests :math:`X \perp Y \mid Z` by computing the
    partial cross-covariance :math:`\hat{C}_{XY \mid Z} = \hat{C}_{XY} -
    \hat{C}_{XZ} \hat{C}_{ZZ}^{-1} \hat{C}_{ZY}` in random feature space.

    The authors recommend RCoT over RCIT as a general-purpose test.

    When Z is empty the test falls back to :class:`Pearsonr`.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.
    num_features_x : int, default=5
        Number of random Fourier features for X.
    num_features_y : int, default=5
        Number of random Fourier features for Y.
    num_features_z : int, default=100
        Number of random Fourier features for Z.
    seed : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    statistic_ : float
        The RCoT test statistic :math:`n \cdot \|\hat{C}_{XY \mid Z}\|_F^2`.
        When Z is empty the fallback :class:`Pearsonr` is used and
        ``statistic_`` holds Pearson's r instead.
        Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import RCoT
    >>> rng = np.random.default_rng(seed=42)
    >>> Z = rng.standard_normal(1000)
    >>> X = Z + rng.standard_normal(1000)
    >>> Y = Z + rng.standard_normal(1000)
    >>> data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    >>> test = RCoT(data=data, seed=0)
    >>> test("X", "Y", ["Z"], significance_level=0.05)
    True

    References
    ----------
    .. [1] Strobl, E. V., Zhang, K., & Visweswaran, S. (2019). Approximate
       Kernel-Based Conditional Independence Tests for Fast Non-Parametric
       Causal Discovery. Statistics and Computing, 29(5), 891-912.
       https://arxiv.org/abs/1702.03877
    """

    _tags = {
        "name": "rcot",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        num_features_x: int = 5,
        num_features_y: int = 5,
        num_features_z: int = 100,
        seed: int | None = None,
    ):
        self.data = data
        self.num_features_x = num_features_x
        self.num_features_y = num_features_y
        self.num_features_z = num_features_z
        self.seed = seed
        super().__init__()

    def run_test(self, X: str, Y: str, Z: list):
        """
        Compute the RCoT statistic and p-value.

        Sets ``self.statistic_`` and ``self.p_value_``.
        """
        data = self.data
        n = data.shape[0]
        rng = np.random.default_rng(self.seed)

        x = data[X].to_numpy(dtype=float)
        y = data[Y].to_numpy(dtype=float)

        # Step 1: Fall back to Pearsonr for the unconditional case.
        if len(Z) == 0:
            from .pearsonr import Pearsonr

            stat, pval = Pearsonr(data=data).run_test(X=X, Y=Y, Z=[])
            self.statistic_ = float(stat)
            self.p_value_ = float(pval)
            return self.statistic_, self.p_value_

        z = data[Z].to_numpy(dtype=float)
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        # Step 2: Normalize x, y, z to zero mean and unit variance.
        x = _normalize(x)
        y = _normalize(y)
        z = _normalize(z)

        # Step 3: Compute bandwidths via median heuristic on normalized data.
        bw_x = _median_bandwidth(x)
        bw_y = _median_bandwidth(y)
        bw_z = _median_bandwidth(z)

        # Step 4: Compute normalized random Fourier features.
        # RCoT uses y directly — no augmentation with z.
        f_x = _rff(x, self.num_features_x, bw_x, rng)
        f_y = _rff(y, self.num_features_y, bw_y, rng)
        f_z = _rff(z, self.num_features_z, bw_z, rng)

        # Step 5: Partial cross-covariance and residuals via covariance projection.
        C_xy_z, res_x, res_y = _partial_covariance(f_x, f_y, f_z)

        # Step 6: Test statistic.
        statistic = float(n * np.sum(C_xy_z**2))

        # Step 7: Eigenvalues of residual cross-product covariance for the null.
        lambdas = _null_eigenvalues(res_x, res_y)

        # Step 8: P-value via Gamma approximation.
        p_value = _gamma_pvalue(statistic, lambdas)

        self.statistic_ = statistic
        self.p_value_ = max(0.0, p_value)
        return self.statistic_, self.p_value_
