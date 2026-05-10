import numpy as np

from pgmpy.structure_score._base import BaseStructureScore


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

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.
    state_names : dict, optional
        Accepted for API consistency but not typically used for Gaussian networks.

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

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)
        self._np_data = self.data.to_numpy()
        self._col_index = {col: i for i, col in enumerate(self.data.columns)}
        self._n_samples = self._np_data.shape[0]
        self._ll_const = -0.5 * self._n_samples * (np.log(2.0 * np.pi) + 1.0)

    def _log_likelihood(self, variable: str, parents: tuple[str, ...]) -> tuple[float, float]:
        n = self._n_samples
        y = self._np_data[:, self._col_index[variable]]

        if len(parents) == 0:
            resid = y - y.mean()
            df_model = 0
        else:
            # Create the covariate matrix
            parent_cols = [self._col_index[p] for p in parents]
            X = np.empty((n, len(parents) + 1))
            X[:, 0] = 1.0
            X[:, 1:] = self._np_data[:, parent_cols]

            # Fit a OLS and compute residuals.
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            df_model = len(parents)

        rss = float(resid @ resid)
        ll = self._ll_const - 0.5 * n * np.log(rss / n)
        return (ll, df_model)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        ll, _ = self._log_likelihood(variable=variable, parents=parents)

        return ll
