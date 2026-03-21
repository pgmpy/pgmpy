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

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def _log_likelihood(self, variable: str, parents: tuple[str, ...]) -> tuple[float, float]:
        parents = self._validate_parents(parents)
        if len(parents) == 0:
            glm_model = smf.glm(formula=f"{variable} ~ 1", data=self.data).fit()
        else:
            glm_model = smf.glm(formula=f"{variable} ~ {' + '.join(parents)}", data=self.data).fit()

        return (glm_model.llf, glm_model.df_model)

    def local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """Compute the local Gaussian log-likelihood score for `variable`."""
        ll, _ = self._log_likelihood(variable=variable, parents=parents)

        return ll
