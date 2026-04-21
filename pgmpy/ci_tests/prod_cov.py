import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

from pgmpy.utils import preprocess_data

from ._base import _BaseCITest, _ResidualMixin


class ProdCov(_ResidualMixin, _BaseCITest):
    r"""
    Product of squared canonical correlations CI test with permutation-based p-values [1].

    This test first residualizes :math:`X` and :math:`Y` with respect to :math:`[1, Z]` using
    an estimator (see :class:`~pgmpy.ci_tests.PillaiTrace` for residualization details). The
    test statistic is the product of all squared canonical correlations between the residuals:

    .. math::
        T = \prod_{k=1}^{s} \hat{\rho}_k^2,

    where :math:`\hat{\rho}_k` are the canonical correlations between :math:`R_X` and
    :math:`R_Y`, and :math:`s = \min(p, q)` with :math:`p` and :math:`q` the number of
    columns in the respective residual matrices.

    :math:`T` equals zero when :math:`X \perp\!\!\!\perp Y \mid Z` (all canonical correlations
    are zero) and is positive under dependence. Because no closed-form null distribution
    is available, the p-value is computed by a permutation test: rows of :math:`R_X` are
    shuffled repeatedly and :math:`T` is recomputed; the p-value is the proportion of
    permuted statistics that meet or exceed the observed value (with a +1 continuity
    correction following Phipson & Smyth [2]).

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

    random_state : int, RandomState instance, or None, optional
        Seed or random state for the permutation sampling. Pass an integer for
        reproducible results.

    Attributes
    ----------
    statistic_ : float
        The observed product of squared canonical correlations :math:`T`. Set after
        calling the test.
    p_value_ : float
        Permutation-based p-value. Set after calling the test.

    References
    ----------
    .. [1] Ankan, Ankur, and Johannes Textor. "A simple unified approach to testing
           high-dimensional conditional independences for categorical and ordinal data."
           Proceedings of the AAAI Conference on Artificial Intelligence.
    .. [2] Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be zero:
           calculating exact p-values when permutations are randomly drawn. Statistical
           Applications in Genetics and Molecular Biology, 9(1).
    """

    _tags = {
        "name": "prod_cov",
        "data_types": ("discrete", "continuous", "mixed"),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        estimator=None,
        n_permutations: int = 1000,
        random_state=None,
    ):
        self.data, self.dtypes = preprocess_data(data)
        self.estimator = estimator
        self.n_permutations = n_permutations
        self.random_state = random_state
        super().__init__()

    def run_test(self, X: str, Y: str, Z: list):
        """
        Compute the product of squared canonical correlations and its permutation p-value.

        Sets ``self.statistic_`` (product of squared canonical correlations) and
        ``self.p_value_``.

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
            The product of squared canonical correlations.
        p_value : float
            Permutation-based p-value.
        """
        # Step 1: Compute residuals of X and Y given Z.
        res_x = self.get_residuals(X, Z)
        res_y = self.get_residuals(Y, Z)

        if isinstance(res_x, pd.Series):
            res_x = res_x.to_frame()
        if isinstance(res_y, pd.Series):
            res_y = res_y.to_frame()

        # Step 2: Compute squared canonical correlations via CCA.
        s = min(res_x.shape[1], res_y.shape[1])
        cca = CCA(scale=False, n_components=s)
        res_x_c, res_y_c = cca.fit_transform(res_x, res_y)

        cancor2 = np.array([np.corrcoef(res_x_c[:, i], res_y_c[:, i])[0, 1] ** 2 for i in range(s)])

        statistic = float(np.prod(cancor2))

        # Step 3: Permutation test to obtain the null distribution of T.
        rng = np.random.default_rng(self.random_state)
        perm_stats = np.empty(self.n_permutations)
        for t in range(self.n_permutations):
            idx = rng.permutation(res_x.shape[0])
            res_x_perm = res_x.iloc[idx].reset_index(drop=True)
            cca_perm = CCA(scale=False, n_components=s)
            rxc_p, ryc_p = cca_perm.fit_transform(res_x_perm, res_y)
            cc2_p = np.array([np.corrcoef(rxc_p[:, i], ryc_p[:, i])[0, 1] ** 2 for i in range(s)])
            perm_stats[t] = float(np.prod(cc2_p))

        # Phipson-Smyth +1 correction so that p-value is never exactly zero.
        p_value = (float(np.sum(perm_stats >= statistic)) + 1.0) / (self.n_permutations + 1.0)

        self.statistic_ = statistic
        self.p_value_ = p_value
        return self.statistic_, self.p_value_
