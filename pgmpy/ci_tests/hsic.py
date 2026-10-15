import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF, Kernel

from ._base import BaseCITest, _CITestResult


class HSIC(BaseCITest):
    r"""
    Test whether two continuous variables are independent.

    HSIC can detect both linear and nonlinear relationships. It checks whether observations that are similar in X also
    tend to be similar in Y.

    Calling the test returns ``True`` when there is not enough evidence to reject independence, and ``False`` when the
    data indicate a relationship. HSIC does not support conditioning variables; use :class:`KCI` for that.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the variables to test. Rows are observations and columns are variables.

    kernel : sklearn.gaussian_process.kernels.Kernel or dict of {str: Kernel}, default=None
        Kernel used to compare observations. A single kernel is used for every column, while a dict maps column names
        to their kernels. Columns missing from the dict, or every column if None, use a default RBF kernel.

    bandwidth : {"heuristic", "median"}, default="heuristic"
        Determines the width of the default RBF kernels.

    null_dist : {"gamma", "permutation"}, default="gamma"
        How to compute the p-value. ``"gamma"`` is fast and works well with RBF-like kernels. ``"permutation"`` is
        slower but works with arbitrary kernels.

    use_cache : bool, default=True
        Whether to cache test results for repeated queries.

    n_permutations : int, default=100
        Used only when ``null_dist="permutation"``. Number of shuffles used for a permutation p-value.

    seed : int or None, default=None
        Used only when ``null_dist="permutation"``. Seed for reproducible permutations.

    Examples
    --------
    >>> from pgmpy.datasets import load_dataset
    >>> from pgmpy.ci_tests import HSIC
    >>> data = load_dataset("tubingen/1").data
    >>> test = HSIC(data=data)

    >>> test("x", "y", significance_level=0.05)
    False
    >>> round(test.statistic_, 2)
    5468.71
    >>> f"{test.p_value_:.2e}"
    '2.97e-67'

    Attributes
    ----------
    statistic_ : float
        The HSIC test statistic. Larger values indicate stronger evidence of a relationship, but values should not be
        compared across sample sizes.

    p_value_ : float
        The p-value for the test. Set after calling the test.

    effect_size_ : None
        Not defined for HSIC.

    Raises
    ------
    ValueError
        If a requested variable is constant, conditioning variables are supplied, or an initialization argument is
        invalid.

    Notes
    -----
    The variables are standardized before testing. With the default RBF kernels, ``bandwidth="heuristic"`` chooses a
    width from the sample size, while ``bandwidth="median"`` uses the median distance between observations.

    The statistic is computed from centered kernel matrices:

    .. math::
        T = \operatorname{Tr}(\tilde{K}_X \tilde{K}_Y).

    When `null_dist='gamma'`, p-value is approximated using a Gamma distribution. When `null_dist='permutation'`,
    p-value is approximated empirically by a permutation test.

    Approximate p-value requires at least six observations; for smaller datasets, ``p_value_`` is set to ``1.0``.

    References
    ----------
    - :footcite:t:`gretton_2005`
    - :footcite:t:`gretton_2007`
    """

    _tags = {
        "name": "hsic",
        "data_types": ["continuous"],
        "default_for": None,
        "requires_data": True,
        "is_symmetric": True,
        "capability:conditional_ci": False,
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        kernel: Kernel | dict[str, Kernel] | None = None,
        bandwidth: str = "heuristic",
        null_dist: str = "gamma",
        n_permutations: int = 100,
        seed: int | None = None,
        use_cache: bool = True,
    ):
        # Step 0: Validate arguments
        if bandwidth not in ("heuristic", "median"):
            raise ValueError(f"bandwidth must be 'heuristic' or 'median', got {bandwidth!r}")
        if null_dist not in ("gamma", "permutation"):
            raise ValueError(f"null_dist must be 'gamma' or 'permutation', got {null_dist!r}")
        if n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1, got {n_permutations!r}")

        # Step 1: Store attributes.
        self.data = data
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.null_dist = null_dist
        self.n_permutations = n_permutations
        self.seed = seed

        # Step 2: Standardize the data.
        self._std_data = self.data.apply(lambda col: (col - col.mean()) / col.std(ddof=1))

        # Step 3: Set up a kernel for each column.
        if isinstance(kernel, Kernel):
            # Step 3.1: If a single kernel is provided, use it for every column.
            self.kernel_dict = dict.fromkeys(self.data.columns, kernel)
        elif kernel is None or isinstance(kernel, dict):
            # Step 3.2: Use the provided per-column kernels; the remaining columns get an RBF kernel.
            kernel = {} if kernel is None else kernel
            invalid = [col for col, k in kernel.items() if col not in self.data.columns or not isinstance(k, Kernel)]
            if invalid:
                raise ValueError(f"kernel must map columns in data to sklearn Kernels. Invalid entries: {invalid}")

            n = self.data.shape[0]
            self.kernel_dict = {}
            for col, x in self._std_data.items():
                if col in kernel:
                    self.kernel_dict[col] = kernel[col]
                    continue
                if self.bandwidth == "median":
                    med = np.median(pdist(x.to_numpy().reshape(-1, 1), metric="euclidean"))
                    length_scale = np.sqrt(2.0) * med if med > 0 else 1.0
                else:
                    length_scale = 0.8 if n < 200 else (0.5 if n < 1200 else 0.3)
                self.kernel_dict[col] = RBF(length_scale=length_scale)
        else:
            raise ValueError(
                f"kernel must be a sklearn Kernel, a dict of Kernels keyed by column, or None. Got {kernel!r}"
            )

        super().__init__(use_cache=use_cache)

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
        x, y = self._std_data[[X]].to_numpy(), self._std_data[[Y]].to_numpy()
        kernel_x, kernel_y = self.kernel_dict[X], self.kernel_dict[Y]

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
