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
        idx = np.random.default_rng(0).choice(n, 500, replace=False)
        x = x[idx]
    dists_sq = np.sum((x[:, None] - x[None, :]) ** 2, axis=-1)
    upper = dists_sq[np.triu_indices_from(dists_sq, k=1)]
    if upper.size == 0 or upper.max() == 0:
        return 1.0
    return float(np.sqrt(np.median(upper)))


def _rff(x: np.ndarray, num_features: int, bandwidth: float, rng: np.random.Generator) -> np.ndarray:
    """
    Compute centered random Fourier features for a Gaussian kernel.

    Uses the cosine approximation: phi(x) = sqrt(2/r) * cos(omega^T x + b)
    where omega ~ N(0, I/bandwidth^2) and b ~ Uniform(0, 2*pi).
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    d = x.shape[1]
    omega = rng.standard_normal((d, num_features)) / bandwidth
    b = rng.uniform(0, 2 * np.pi, num_features)
    phi = np.sqrt(2.0 / num_features) * np.cos(x @ omega + b)
    phi -= phi.mean(axis=0)
    return phi


def _residuals(phi: np.ndarray, phi_z: np.ndarray) -> np.ndarray:
    """Regress columns of phi on phi_z and return the residuals."""
    coef, _, _, _ = np.linalg.lstsq(phi_z, phi, rcond=None)
    return phi - phi_z @ coef


def _gamma_pvalue(statistic: float, lambdas: np.ndarray) -> float:
    """
    P-value via the Satterthwaite/Pearson Gamma approximation for a
    weighted chi-squared null distribution.

    Under H0, the test statistic is distributed as sum_i lambda_i * chi2(1).
    This is approximated by c * chi2(k) where:
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
    Fourier Features (RFF) for X, Y, and Z. It tests :math:`X \perp Y \mid Z`
    by projecting RFF maps of X and Y onto the orthogonal complement of the RFF
    map of Z, then measuring the residual cross-covariance.

    Under the null hypothesis :math:`X \perp Y \mid Z`, the scaled test
    statistic :math:`n \cdot \|\hat{C}_{XY \mid Z}\|_F^2` follows an
    approximate weighted :math:`\chi^2` distribution. The p-value is obtained
    via a Satterthwaite/Gamma approximation using the eigenvalues of
    :math:`\hat{C}_{XX \mid Z} \otimes \hat{C}_{YY \mid Z}`.

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
        The RCIT test statistic :math:`n \cdot \|\hat{C}_{XY \mid Z}\|_F^2`.
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

        # Step 2: Compute bandwidths via median heuristic.
        bw_x = _median_bandwidth(x)
        bw_y = _median_bandwidth(y)
        bw_z = _median_bandwidth(z)

        # Step 3: Compute random Fourier features.
        phi_x = _rff(x, self.num_features_x, bw_x, rng)
        phi_y = _rff(y, self.num_features_y, bw_y, rng)
        phi_z = _rff(z, self.num_features_z, bw_z, rng)

        # Step 4: Project out Z from X- and Y-features.
        r_x = _residuals(phi_x, phi_z)
        r_y = _residuals(phi_y, phi_z)

        # Step 5: Cross-covariance matrix and test statistic.
        c_xy = (r_x.T @ r_y) / n
        statistic = float(n * np.sum(c_xy**2))

        # Step 6: Eigenvalues of Cxx x Cyy for the null distribution.
        c_xx = (r_x.T @ r_x) / n
        c_yy = (r_y.T @ r_y) / n
        eig_xx = np.linalg.eigvalsh(c_xx)
        eig_yy = np.linalg.eigvalsh(c_yy)
        lambdas = np.outer(eig_xx, eig_yy).ravel()

        # Step 7: P-value via Gamma approximation.
        p_value = _gamma_pvalue(statistic, lambdas)

        self.statistic_ = statistic
        self.p_value_ = p_value
        return self.statistic_, self.p_value_


class RCoT(_BaseCITest):
    r"""
    Randomised Conditional Covariance Test (RCoT) for continuous data.

    RCoT is a computationally lighter variant of RCIT. It represents X and Y
    directly as raw (centered) column vectors rather than random feature maps,
    while using Random Fourier Features only for the conditioning set Z. This
    makes RCoT faster than RCIT, at the cost of restricting the feature map
    for X and Y to a single dimension.

    Under the null hypothesis :math:`X \perp Y \mid Z`, the test statistic
    :math:`n \cdot \hat{c}_{XY \mid Z}^2` follows a :math:`\lambda \cdot \chi^2(1)`
    distribution where :math:`\lambda = \hat{c}_{XX \mid Z} \cdot \hat{c}_{YY \mid Z}`.

    When Z is empty the test falls back to :class:`Pearsonr`.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.
    num_features_z : int, default=100
        Number of random Fourier features for Z.
    seed : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    statistic_ : float
        The RCoT test statistic :math:`n \cdot \hat{c}_{XY \mid Z}^2`.
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
        num_features_z: int = 100,
        seed: int | None = None,
    ):
        self.data = data
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

        # Step 2: Bandwidth for Z; center X and Y.
        bw_z = _median_bandwidth(z)
        x_c = (x - x.mean()).reshape(-1, 1)
        y_c = (y - y.mean()).reshape(-1, 1)

        # Step 3: Random Fourier features for Z.
        phi_z = _rff(z, self.num_features_z, bw_z, rng)

        # Step 4: Project out Z from X and Y.
        r_x = _residuals(x_c, phi_z)  # n x 1
        r_y = _residuals(y_c, phi_z)  # n x 1

        # Step 5: Scalar cross-covariance and test statistic.
        c_xy = float((r_x[:, 0] @ r_y[:, 0]) / n)
        statistic = float(n * c_xy**2)

        # Step 6: Null-distribution scale parameter.
        c_xx = float((r_x[:, 0] @ r_x[:, 0]) / n)
        c_yy = float((r_y[:, 0] @ r_y[:, 0]) / n)
        lam = c_xx * c_yy

        # Step 7: P-value: statistic / lam ~ chi2(1) under H0.
        p_value = float(stats.chi2.sf(statistic / lam, df=1)) if lam > 1e-10 else 1.0

        self.statistic_ = statistic
        self.p_value_ = p_value
        return self.statistic_, self.p_value_
