import numpy as np

from pgmpy.base import DAG
from pgmpy.structure_score.log_likelihood_gauss import LogLikelihoodGauss


class BICGauss(LogLikelihoodGauss):
    """BIC structure score for Gaussian Bayesian networks."""

    _tags = {
        "name": "bic_gauss_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": True,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def local_score(self, variable: str, parents: list[str]) -> float:
        """Compute the local Gaussian BIC score for `variable`."""
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        return ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))
