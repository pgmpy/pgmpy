import numpy as np

from pgmpy.structure_score.log_likelihood_cond_gauss import LogLikelihoodCondGauss


class BICCondGauss(LogLikelihoodCondGauss):
    """BIC structure score for mixed Bayesian networks."""

    _tags = {
        "name": "bic-cg",
        "supported_datatype": "mixed",
        "default_for": "mixed",
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        r"""
        Compute the local mixed-data BIC score for ``variable`` given ``parents``.

        The method computes

        .. math::
            \operatorname{BIC}(X_i, \Pi_i)
            = \ell(X_i, \Pi_i) - \frac{k_i}{2} \log n,

        where :math:`\ell(X_i, \Pi_i)` is the local
        conditional-Gaussian log-likelihood, :math:`k_i` is the number of free
        parameters computed by ``_get_num_parameters``, and :math:`n` is the
        number of rows in ``self.data``.
        """
        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - ((k / 2) * np.log(self.data.shape[0]))
