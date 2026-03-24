from math import lgamma

import numpy as np
from scipy.special import gammaln

from pgmpy.structure_score._base import BaseStructureScore


class K2(BaseStructureScore):
    """K2 structure score for discrete Bayesian networks."""

    _tags = {
        "name": "k2",
        "supported_datatype": "discrete",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        r"""
        Compute the local K2 score for ``variable`` given ``parents``.

        The method computes:

        .. math::
            \operatorname{K2}(X_i, \Pi_i) = \sum_{j=1}^{q_i} \left[ \log \Gamma(r_i)
                - \log \Gamma(N_{ij} + r_i) + \sum_{k=1}^{r_i} \log \Gamma(N_{ijk} + 1) \right],

        where :math:`r_i` is the cardinality of :math:`X_i`, :math:`q_i` is the number of parent configurations of
        :math:`\Pi_i`, :math:`N_{ijk}` is the count of :math:`X_i = k` in parent configuration :math:`j`, and
        :math:`N_{ij} = \sum_{k=1}^{r_i} N_{ijk}`.
        """
        var_cardinality = len(self.state_names[variable])
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_gamma_counts = np.zeros_like(counts, dtype=float)

        gammaln(counts + 1, out=log_gamma_counts)

        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + var_cardinality, out=log_gamma_conds)

        score = np.sum(log_gamma_counts) - np.sum(log_gamma_conds) + num_parents_states * lgamma(var_cardinality)

        return score
