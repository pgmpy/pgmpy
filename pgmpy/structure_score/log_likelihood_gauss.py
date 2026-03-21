import statsmodels.formula.api as smf

from pgmpy.base import DAG
from pgmpy.structure_score.base import BaseStructureScore


class LogLikelihoodGauss(BaseStructureScore):
    """Gaussian log-likelihood structure score."""

    _tags = {
        "name": "log_likelihood_gauss_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def _log_likelihood(self, variable: str, parents: list[str]) -> tuple[float, float]:
        if len(parents) == 0:
            glm_model = smf.glm(formula=f"{variable} ~ 1", data=self.data).fit()
        else:
            glm_model = smf.glm(formula=f"{variable} ~ {' + '.join(parents)}", data=self.data).fit()

        return (glm_model.llf, glm_model.df_model)

    def local_score(self, variable: str, parents: list[str]) -> float:
        """Compute the local Gaussian log-likelihood score for `variable`."""
        ll, _ = self._log_likelihood(variable=variable, parents=parents)

        return ll
