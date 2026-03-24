import statsmodels.formula.api as smf

from pgmpy.structure_score._base import BaseStructureScore


class LogLikelihoodGauss(BaseStructureScore):
    """Gaussian log-likelihood structure score."""

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
        r"""
        Compute the local Gaussian log-likelihood score for ``variable`` given ``parents``.

        The method fits the Gaussian GLM:

        .. math::
            X_i = \beta_0 + \beta^\top \Pi_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2),

        and returns the fitted log-likelihood:

        .. math::
            \ell(X_i, \Pi_i) = \log p(x_i \mid \hat{\beta}_0, \hat{\beta}, \hat{\sigma}_i^2, \Pi_i).

        If ``parents`` is empty, the fitted model reduces to :math:`X_i = \beta_0 + \varepsilon_i`.
        """
        ll, _ = self._log_likelihood(variable=variable, parents=parents)

        return ll
