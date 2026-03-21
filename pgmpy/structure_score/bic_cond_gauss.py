import numpy as np

from pgmpy.base import DAG
from pgmpy.structure_score.log_likelihood_cond_gauss import LogLikelihoodCondGauss


class BICCondGauss(LogLikelihoodCondGauss):
    """BIC structure score for mixed Bayesian networks."""

    _tags = {
        "name": "bic_cond_gauss_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": True,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def local_score(self, variable: str, parents: list[str]) -> float:
        """Compute the local mixed-data BIC score for `variable`."""
        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - ((k / 2) * np.log(self.data.shape[0]))
