from math import lgamma

import numpy as np
from scipy.special import gammaln

from pgmpy.structure_score._base import BaseStructureScore


class BDeu(BaseStructureScore):
    r"""
    BDeu structure score for discrete Bayesian networks with Dirichlet priors.

    The BDeu score evaluates a Bayesian network structure on fully discrete data using a Dirichlet prior parameterized
    by an equivalent sample size. The local score computed as:

    .. math::
        \operatorname{BDeu}(X_i, \Pi_i) = \sum_{j=1}^{q_i} \left[
            \log \Gamma\left(\frac{\alpha}{q_i}\right)
            - \log \Gamma\left(N_{ij} + \frac{\alpha}{q_i}\right)
            + \sum_{k=1}^{r_i} \left(
                \log \Gamma\left(N_{ijk} + \frac{\alpha}{r_i q_i}\right)
                - \log \Gamma\left(\frac{\alpha}{r_i q_i}\right)
            \right)
        \right],

    where :math:`\alpha` is `equivalent_sample_size`, :math:`r_i` is the cardinality of :math:`X_i`, :math:`q_i` is the
    number of parent configurations of :math:`\Pi_i`, :math:`N_{ijk}` is the count of :math:`X_i = k` in parent
    configuration :math:`j`, and :math:`N_{ij} = \sum_{k=1}^{r_i} N_{ijk}`.

    In the implementation, `state_counts(..., reindex=False)` drops unobserved parent configurations to save memory. The
    `gamma_counts_adj` and `gamma_conds_adj` terms restore the missing :math:`\log \Gamma(\beta)` and :math:`\log
    \Gamma(\alpha)` contributions so that the returned value still equals the full BDeu score over all parent
    configurations.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable. Missing values should be
        set to `numpy.nan`.
    equivalent_sample_size : int, optional
        Equivalent sample size used to define the Dirichlet hyperparameters.
    state_names : dict, optional
        Dictionary mapping each variable to its discrete states. If not specified, the unique
        values observed in the data are used.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.structure_score import BDeu
    >>> data = pd.DataFrame(
    ...     {"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]}
    ... )
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> score = BDeu(data, equivalent_sample_size=5)
    >>> round(score.score(model), 3)
    np.float64(-9.392)
    >>> round(score.local_score("B", ("A",)), 3)
    np.float64(-3.446)

    Raises
    ------
    ValueError
        If the data contains non-discrete variables, or if the model variables are not present
        in the data.

    References
    ----------
    .. [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009, Section 18.3.4-18.3.6.
    .. [2] AM Carvalho, Scoring functions for learning Bayesian networks,
        http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

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
