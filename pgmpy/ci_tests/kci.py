import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.gaussian_process.kernels import RBF, Kernel

from ._base import _CITestResult
from .hsic import HSIC, _gamma_pvalue_from_moments


class KCI(HSIC):
    r"""
    Test whether two continuous variables are independent given a set of other variables.

    KCI can detect both linear and nonlinear relationships. Calling the test returns ``True`` when there is not enough
    evidence to reject conditional independence, and ``False`` when a relationship remains after accounting for Z.
    Without conditioning variables, KCI runs :class:`HSIC`.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the variables to test. Rows are observations and columns are variables.

    kernel : sklearn.gaussian_process.kernels.Kernel or dict of {str: Kernel}, default=None
        Kernel used to compare observations. A single kernel is used for X, Y, and the conditioning set. A dict maps
        column names to the kernels used when that column is tested as X or Y; the conditioning set, and columns
        missing from the dict, use a default RBF kernel. If None, every kernel is a default RBF kernel.

    bandwidth : {"heuristic", "median"}, default="heuristic"
        Determines the width of the default RBF kernels.

    epsilon : float, default=1e-3
        Regularization used when removing the effect of the conditioning set. Must be finite and positive.

    use_cache : bool, default=True
        Whether to cache test results for repeated queries.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import KCI
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(rng.standard_normal((300, 3)), columns=["X", "Y", "Z"])
    >>> test = KCI(data=data)

    >>> test("X", "Y", ["Z"], significance_level=0.05)
    True
    >>> round(test.statistic_, 2)
    95.19
    >>> round(test.p_value_, 2)
    0.33
    >>> round(test.effect_size_, 3)
    0.016

    Attributes
    ----------
    statistic_ : float
        The KCI test statistic. Larger values indicate stronger evidence of a remaining relationship, but values should
        not be compared across sample sizes.

    p_value_ : float
        The p-value for the test. Set after calling the test.

    effect_size_ : float
        Normalized KCI statistic, between 0 and 1. Set after calling the test.

    Raises
    ------
    ValueError
        If an initialization argument is invalid.

    Notes
    -----
    The variables are standardized before testing. Following :footcite:t:`zhang_2011_kci`, X is augmented with
    :math:`0.5 Z`, and the centered kernel matrices of X and Y are stripped of the part explained by Z with
    :math:`R_Z = \epsilon (\tilde{K}_Z + \epsilon I)^{-1}`:

    .. math::
        \tilde{K}_{\ddot{X}|Z} = R_Z \tilde{K}_{\ddot{X}} R_Z, \qquad \tilde{K}_{Y|Z} = R_Z \tilde{K}_Y R_Z, \qquad
        T_{CI} = \operatorname{Tr}(\tilde{K}_{\ddot{X}|Z} \tilde{K}_{Y|Z}).

    The p-value comes from a Gamma approximation of the distribution of :math:`T_{CI}` under conditional independence.

    With the default RBF kernels, ``bandwidth="heuristic"`` uses the length scale :math:`w \sqrt{|Z|}` for every
    kernel, where :math:`w` is 1.2, 0.7, or 0.4 for fewer than 200, fewer than 1200, or more observations.
    ``bandwidth="median"`` gives each kernel the median non-zero pairwise distance of its own input divided by
    :math:`\sqrt{2}`, as in :class:`HSIC`.

    The effect size normalizes the statistic, as centered kernel alignment :footcite:t:`cortes_2012` does for HSIC:

    .. math::
        \frac{\operatorname{Tr}(\tilde{K}_{\ddot{X}|Z} \tilde{K}_{Y|Z})}
        {\sqrt{\operatorname{Tr}(\tilde{K}_{\ddot{X}|Z}^2) \operatorname{Tr}(\tilde{K}_{Y|Z}^2)}}.

    It lies between 0 and 1 and matches the squared partial correlation of X and Y given Z when all kernels are linear.
    With RBF kernels the values are small and biased upwards in small samples, so compare them only at the same sample
    size.

    References
    ----------
    - :footcite:t:`zhang_2011_kci`
    - :footcite:t:`cortes_2012`
    """

    _tags = {
        "name": "kci",
        "data_types": ["continuous"],
        "default_for": None,
        "requires_data": True,
        "is_symmetric": True,
        "capability:conditioning": True,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        kernel: Kernel | dict[str, Kernel] | None = None,
        bandwidth: str = "heuristic",
        epsilon: float = 1e-3,
        use_cache: bool = True,
    ):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(f"epsilon must be a finite positive number, got {epsilon!r}")

        self.epsilon = epsilon
        super().__init__(data=data, kernel=kernel, bandwidth=bandwidth, use_cache=use_cache)

    def _compute_result(self, X: str, Y: str, Z: list):
        r"""
        Compute KCI statistic, p-value, and effect size for :math:`X \perp Y \mid Z`; uses HSIC when Z is empty.
        """
        if not Z:
            return super()._compute_result(X, Y, Z)

        x_aug = np.hstack([self._std_data[[X]].to_numpy(), 0.5 * self._std_data[Z].to_numpy()])
        y = self._std_data[[Y]].to_numpy()
        z = self._std_data[Z].to_numpy()

        # Step 1: Set up the kernels.
        if isinstance(self.kernel, Kernel):
            kernel_x = kernel_y = kernel_z = self.kernel
        else:
            kernels = self.kernel or {}
            if self.bandwidth == "median":
                ls_x, ls_y, ls_z = (self._median_length_scale(a) for a in (x_aug, y, z))
            else:
                ls_x = ls_y = ls_z = (
                    1.2 if self._n_samples < 200 else (0.7 if self._n_samples < 1200 else 0.4)
                ) * np.sqrt(len(Z))
            kernel_x = kernels[X] if X in kernels else RBF(length_scale=ls_x)
            kernel_y = kernels[Y] if Y in kernels else RBF(length_scale=ls_y)
            kernel_z = RBF(length_scale=ls_z)

        # Step 2: Remove the part of the X and Y kernels explained by Z.
        Kx = self._center_kernel(kernel_x(x_aug))
        Ky = self._center_kernel(kernel_y(y))
        Kz = self._center_kernel(kernel_z(z))
        identity = np.eye(self._n_samples)
        Rz = self.epsilon * linalg.solve(Kz + self.epsilon * identity, identity, assume_a="pos")
        KxR = Rz @ Kx @ Rz
        KyR = Rz @ Ky @ Rz

        test_stat = np.vdot(KxR, KyR)
        effect_size = test_stat / np.sqrt(np.vdot(KxR, KxR) * np.vdot(KyR, KyR))

        # Step 3: Gamma approximation of the null distribution from the eigenvectors of KxR and KyR.
        wx, vx = np.linalg.eigh(0.5 * (KxR + KxR.T))
        wy, vy = np.linalg.eigh(0.5 * (KyR + KyR.T))
        keep_x, keep_y = wx > wx.max() * 1e-5, wy > wy.max() * 1e-5
        vx = vx[:, keep_x] * np.sqrt(wx[keep_x])
        vy = vy[:, keep_y] * np.sqrt(wy[keep_y])
        uu = (vx[:, :, None] * vy[:, None, :]).reshape(self._n_samples, -1)
        uu_prod = uu @ uu.T if uu.shape[1] > self._n_samples else uu.T @ uu
        p_value = _gamma_pvalue_from_moments(test_stat, np.trace(uu_prod), 2.0 * np.vdot(uu_prod, uu_prod))

        return _CITestResult(statistic=float(test_stat), p_value=float(p_value), effect_size=float(effect_size))
