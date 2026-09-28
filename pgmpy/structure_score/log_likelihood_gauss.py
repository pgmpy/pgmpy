from collections.abc import Hashable

import numpy as np

from pgmpy.structure_score._base import BaseStructureScore
from pgmpy.utils import _check_no_missing_values, covariance_sufficient_stats, residual_covariance


class LogLikelihoodGauss(BaseStructureScore):
    r"""
    Log-likelihood structure score for Gaussian Bayesian networks.

    This score evaluates a continuous Bayesian network structure by fitting a Gaussian GLM for each local family and
    returning the fitted log-likelihood. The local score is computed as:

    .. math::
        X_i = \beta_0 + \beta^\top \Pi_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2),

    and returns

    .. math::
        \ell(X_i, \Pi_i) = \log p(x_i \mid \hat{\beta}_0, \hat{\beta}, \hat{\sigma}_i^2, \Pi_i).

    If `parents` is empty, the fitted model reduces to :math:`X_i = \beta_0 + \varepsilon_i`.

    A family whose parents determine :math:`X_i` to within floating point precision scores
    :math:`-\infty`, as in bnlearn.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.
    state_names : dict, optional
        Accepted for API consistency but not typically used for Gaussian networks.
    max_cache_size : int or None, default=10000
        Maximum number of local scores to cache. If None, the cache is unlimited.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.structure_score import LogLikelihoodGauss
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": rng.normal(size=100),
    ...         "B": rng.normal(size=100),
    ...         "C": rng.normal(size=100),
    ...     }
    ... )
    >>> score = LogLikelihoodGauss(data)
    >>> round(score.local_score("B", ("A", "C")), 3)
    np.float64(-137.16)

    Raises
    ------
    ValueError
        If the model cannot be fitted because the data contains incompatible or non-numeric variables.
    """

    _tags = {
        "name": "ll-g",
        "supported_datatype": "continuous",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None, max_cache_size=10000):
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)
        self._cov, _, self._col_index, self._missing = covariance_sufficient_stats(self.data)
        self._n_samples = self.data.shape[0]
        self._ll_const = -0.5 * self._n_samples * (np.log(2.0 * np.pi) + 1.0)

    def _log_likelihood(self, variable: Hashable, parents: tuple[Hashable, ...]) -> tuple[float, float]:
        _check_no_missing_values((variable, *parents), self._missing, f"The Gaussian log-likelihood of {variable}")
        variable_col = self._col_index[variable]
        parent_cols = [self._col_index[parent] for parent in parents]
        residual_var = residual_covariance(self._cov, [variable_col], parent_cols)[0, 0]
        if residual_var <= np.sqrt(np.finfo(float).eps) * self._cov[variable_col, variable_col]:
            return (-np.inf, len(parents))

        ll = self._ll_const - 0.5 * self._n_samples * np.log(residual_var)
        return (ll, len(parents))

    def _local_score(self, variable: Hashable, parents: tuple[Hashable, ...]) -> float:
        ll, _ = self._log_likelihood(variable=variable, parents=parents)

        return ll
