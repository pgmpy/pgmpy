import numpy as np

from pgmpy.structure_score.log_likelihood_gauss import LogLikelihoodGauss


class BICGauss(LogLikelihoodGauss):
    """BIC structure score for Gaussian Bayesian networks."""

    _tags = {
        "name": "bic-g",
        "supported_datatype": "continuous",
        "default_for": "continuous",
        "is_parameteric": False,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """Compute the local Gaussian BIC score for `variable`."""
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        return ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))
