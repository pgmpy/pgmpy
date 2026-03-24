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

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        r"""
        Compute the local Gaussian BIC score for ``variable`` given ``parents``.

        The method computes

        .. math::
            \operatorname{BIC}(X_i, \Pi_i)
            = \ell(X_i, \Pi_i) - \frac{d_i}{2} \log n,

        where :math:`\ell(X_i, \Pi_i)` is the fitted Gaussian
        log-likelihood, :math:`d_i = \text{df\_model} + 2` is the
        effective parameter count used by the implementation, and
        :math:`n` is the number of rows in ``self.data``.
        """
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        return ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))
