from math import lgamma

import numpy as np
from scipy.special import gammaln

from pgmpy.structure_score._base import BaseStructureScore


class BDeu(BaseStructureScore):
    """BDeu structure score for discrete Bayesian networks."""

    _tags = {
        "name": "bdeu",
        "supported_datatype": "discrete",
        "default_for": None,
        "is_parameteric": True,
    }

    def __init__(self, data, equivalent_sample_size=10, state_names=None):
        self.equivalent_sample_size = equivalent_sample_size
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        r"""
        Compute the local BDeu score for ``variable`` given ``parents``.

        The method computes

        .. math::
            \operatorname{BDeu}(X_i, \Pi_i)
            = \sum_{j=1}^{q_i} \left[
                \log \Gamma\left(\frac{\alpha}{q_i}\right)
                - \log \Gamma\left(N_{ij} + \frac{\alpha}{q_i}\right)
                + \sum_{k=1}^{r_i}
                  \left(
                    \log \Gamma\left(N_{ijk} + \frac{\alpha}{r_i q_i}\right)
                    - \log \Gamma\left(\frac{\alpha}{r_i q_i}\right)
                  )
              \right],

        where :math:`\alpha` is ``equivalent_sample_size``, :math:`r_i` is
        the cardinality of :math:`X_i`, :math:`q_i` is the number of parent
        configurations of :math:`\Pi_i`, :math:`N_{ijk}` is the count of
        :math:`X_i = k` in parent configuration :math:`j`, and
        :math:`N_{ij} = \sum_{k=1}^{r_i} N_{ijk}`.
        """
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        counts_size = num_parents_states * len(self.state_names[variable])
        log_gamma_counts = np.zeros_like(counts, dtype=float)
        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts_size
        gammaln(counts + beta, out=log_gamma_counts)

        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        gamma_counts_adj = (num_parents_states - counts.shape[1]) * len(self.state_names[variable]) * gammaln(beta)
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(alpha)

        score = (
            (np.sum(log_gamma_counts) + gamma_counts_adj)
            - (np.sum(log_gamma_conds) + gamma_conds_adj)
            + num_parents_states * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return score
