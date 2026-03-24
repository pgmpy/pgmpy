import statsmodels.formula.api as smf

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

    def _log_likelihood(self, variable: str, parents: tuple[str, ...]) -> tuple[float, float]:
        if len(parents) == 0:
            glm_model = smf.glm(formula=f"{variable} ~ 1", data=self.data).fit()
        else:
            glm_model = smf.glm(formula=f"{variable} ~ {' + '.join(parents)}", data=self.data).fit()

        return (glm_model.llf, glm_model.df_model)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        ll, _ = self._log_likelihood(variable=variable, parents=parents)

        return ll
