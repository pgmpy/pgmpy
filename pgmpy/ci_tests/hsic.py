import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF, Kernel

from ._base import BaseCITest, _CITestResult


class HSIC(BaseCITest):
    r"""
    Test whether two continuous variables are independent.

    HSIC can detect both linear and nonlinear relationships. It checks whether
    observations that are similar in X also tend to be similar in Y.

    Calling the test returns ``True`` when there is not enough evidence to
    reject independence, and ``False`` when the data indicate a relationship.
    HSIC does not support conditioning variables; use :class:`KCI` for that.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the variables to test. Rows are observations and
        columns are variables.
    kernel : sklearn.gaussian_process.kernels.Kernel or tuple of Kernel, default=None
        Kernel used to compare observations. A single kernel is used for both
        variables, while a 2-tuple provides one for each. Most users can leave
        this as None to use automatically configured RBF kernels.
    bandwidth : {"heuristic", "median"}, default="heuristic"
        How to choose the RBF kernel width when ``kernel`` is None.
    null_dist : {"gamma", "permutation"}, default="gamma"
        How to compute the p-value. ``"gamma"`` is fast and works well with
        RBF-like kernels. ``"permutation"`` is slower but works with arbitrary
        kernels.
    n_permutations : int, default=100
        Number of shuffles used for a permutation p-value. More shuffles give
        a more precise result but take longer.
    seed : int or None, default=None
        Seed for reproducible permutations. Used only when
        ``null_dist="permutation"``.
    use_cache : bool, default=True
        Whether to cache test results for repeated queries.

    Examples
    --------
    >>> from pgmpy.datasets import load_dataset
    >>> from pgmpy.ci_tests import HSIC
    >>> data = load_dataset("tubingen/1").data
    >>> test = HSIC(data=data)
    >>> # False means that independence is rejected at the 5% level.
    >>> test("x", "y", significance_level=0.05)
    False
    >>> round(test.statistic_, 2)
    5468.71
    >>> f"{test.p_value_:.2e}"
    '2.97e-67'

    Attributes
    ----------
    statistic_ : float
        The HSIC test statistic. Larger values indicate stronger evidence of a
        relationship, but values should not be compared across sample sizes.
    p_value_ : float
        The p-value for the test. Set after calling the test.
    effect_size_ : None
        Not defined for HSIC.

    Raises
    ------
    ValueError
        If a requested variable is constant, conditioning variables are
        supplied, or an initialization argument is invalid.

    Notes
    -----
    The variables are standardized before testing. With the default RBF
    kernels, ``bandwidth="heuristic"`` chooses a width from the sample size,
    while ``bandwidth="median"`` uses the median distance between observations.

    The statistic is computed from centered kernel matrices:

    .. math::
        T = \operatorname{Tr}(\tilde{K}_X \tilde{K}_Y).

    The Gamma method approximates the distribution of this statistic when X
    and Y are independent. It requires at least six observations; for smaller
    datasets, ``p_value_`` is set to ``1.0``. The permutation method instead
    estimates this distribution by repeatedly shuffling Y.

    References
    ----------
    - :footcite:t:`gretton_2005`
    - :footcite:t:`gretton_2007`
    """

    _tags = {
        "name": "hsic",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
        "is_symmetric": True,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        kernel: Kernel | tuple | None = None,
        bandwidth: str = "heuristic",
        null_dist: str = "gamma",
        n_permutations: int = 100,
        seed: int | None = None,
        use_cache: bool = True,
    ):
        if bandwidth not in ("heuristic", "median"):
            raise ValueError(f"bandwidth must be 'heuristic' or 'median', got {bandwidth!r}")
        if null_dist not in ("gamma", "permutation"):
            raise ValueError(f"null_dist must be 'gamma' or 'permutation', got {null_dist!r}")
        if n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1, got {n_permutations!r}")

        if kernel is None or isinstance(kernel, Kernel):
            self.kernel_X_ = self.kernel_Y_ = kernel
        elif isinstance(kernel, tuple) and len(kernel) == 2 and all(isinstance(item, Kernel) for item in kernel):
            self.kernel_X_, self.kernel_Y_ = kernel
        else:
            raise ValueError("kernel must be a sklearn Kernel, a tuple of two Kernels, or None")

        self.data = data
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.null_dist = null_dist
        self.n_permutations = n_permutations
        self.seed = seed

        # Standardize each non-constant column once; constant ones are rejected
        # in `_compute_result`. Default RBF kernels are built only if needed.
        std = self.data.std(ddof=1)
        self._std_data = {
            col: ((self.data[col] - self.data[col].mean()) / std[col]).to_numpy().reshape(-1, 1)
            for col in self.data.columns
            if std[col] > 0
        }
        self._default_kernels = (
            {col: RBF(length_scale=self._length_scale(x)) for col, x in self._std_data.items()}
            if self.kernel_X_ is None or self.kernel_Y_ is None
            else {}
        )

        super().__init__(use_cache=use_cache)

    @staticmethod
    def _median_width(X: np.ndarray) -> float:
        """Median heuristic: length-scale = sqrt(2) * median(euclidean_dists)."""
        med = np.median(pdist(X, metric="euclidean"))
        return np.sqrt(2.0) * med if med > 0 else 1.0

    def _length_scale(self, X: np.ndarray) -> float:
        """RBF length-scale: median heuristic, or piecewise width by sample size."""
        if self.bandwidth == "median":
            return self._median_width(X)
        n = X.shape[0]
        width = 0.8 if n < 200 else (0.5 if n < 1200 else 0.3)
        return width / np.sqrt(X.shape[1])

    @staticmethod
    def _center_kernel(K: np.ndarray) -> np.ndarray:
        """Double-center a kernel matrix."""
        col_mean = K.mean(axis=0)
        return K - col_mean[None, :] - col_mean[:, None] + col_mean.mean()

    def _hsic_gamma_pvalue(self, test_stat, K, L, Kc, Lc):
        """Compute a Gamma p-value using the closed-form null moments."""
        n = K.shape[0]
        if n < 6:
            return 1.0

        M_sq = (Kc * Lc / 6.0) ** 2
        off_diag_sq_sum = M_sq.sum() - np.trace(M_sq)
        var_hsic = off_diag_sq_sum / (n * (n - 1))
        moment_factor = 72.0 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))
        var_hsic *= moment_factor

        mu_x = (K.sum() - np.trace(K)) / (n * (n - 1))
        mu_y = (L.sum() - np.trace(L)) / (n * (n - 1))
        mean_hsic = (1.0 + mu_x * mu_y - mu_x - mu_y) / n

        if var_hsic <= 0 or mean_hsic <= 0:
            return 1.0

        alpha = mean_hsic**2 / var_hsic
        beta = var_hsic * n / mean_hsic
        return stats.gamma.sf(test_stat / n, a=alpha, scale=beta)

    def _compute_result(self, X: str, Y: str, Z: list):
        r"""
        Compute HSIC statistic and p-value for :math:`X \perp Y`.

        Z must be empty; use :class:`KCI` for conditional tests.
        """
        if Z:
            raise ValueError("HSIC does not support conditioning variables; use KCI.")

        for var in (X, Y):
            if var not in self._std_data:
                raise ValueError(f"Column {var!r} is constant; HSIC requires non-constant variables.")

        x, y = self._std_data[X], self._std_data[Y]
        kernel_x = self.kernel_X_ if self.kernel_X_ is not None else self._default_kernels[X]
        kernel_y = self.kernel_Y_ if self.kernel_Y_ is not None else self._default_kernels[Y]

        Kx, Ky = kernel_x(x), kernel_y(y)
        Kxc, Kyc = self._center_kernel(Kx), self._center_kernel(Ky)
        test_stat = np.sum(Kxc * Kyc)

        if self.null_dist == "permutation":
            # Permutation commutes with centering, so reindex Kyc instead of re-centering.
            rng = np.random.default_rng(self.seed)
            n = y.shape[0]
            null_stats = np.empty(self.n_permutations)
            for i in range(self.n_permutations):
                perm = rng.permutation(n)
                null_stats[i] = np.sum(Kxc * Kyc[perm][:, perm])
            p_value = (1 + np.sum(null_stats >= test_stat)) / (1 + self.n_permutations)
        else:
            p_value = self._hsic_gamma_pvalue(test_stat, Kx, Ky, Kxc, Kyc)

        return _CITestResult(statistic=float(test_stat), p_value=float(p_value), effect_size=None)
