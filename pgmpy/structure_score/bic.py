from math import log

from pgmpy.structure_score.log_likelihood import LogLikelihood


class BIC(LogLikelihood):
    """BIC structure score for discrete Bayesian networks."""

    _tags = {
        "name": "bic-d",
        "supported_datatype": "discrete",
        "default_for": "discrete",
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        r"""
        Compute the local BIC score for ``variable`` given ``parents``.

        The method computes

        .. math::
            \operatorname{BIC}(X_i, \Pi_i)
            = \ell(X_i, \Pi_i) - \frac{\log n}{2} q_i (r_i - 1),

        where :math:`\ell(X_i, \Pi_i)` is the local discrete log-likelihood,
        :math:`n` is the number of rows in ``self.data``, :math:`q_i` is the
        number of parent configurations of :math:`\Pi_i`, and :math:`r_i` is
        the cardinality of :math:`X_i`.
        """
        sample_size = len(self.data)
        ll, num_parents_states, var_cardinality = self._log_likelihood(variable=variable, parents=parents)
        score = ll - 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)

        return score
