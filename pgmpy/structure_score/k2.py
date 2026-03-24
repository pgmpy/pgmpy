from math import lgamma

import numpy as np
from scipy.special import gammaln

from pgmpy.structure_score._base import BaseStructureScore


class K2(BaseStructureScore):
    r"""
    K2 structure score for discrete Bayesian networks using uniform Dirichlet priors.

    The K2 score evaluates a Bayesian network structure on fully discrete data under a Dirichlet prior in which all
    pseudo-counts are equal to 1. The local score is computed as:

    .. math::
        \operatorname{K2}(X_i, \Pi_i) = \sum_{j=1}^{q_i} \left[ \log \Gamma(r_i)
            - \log \Gamma(N_{ij} + r_i) + \sum_{k=1}^{r_i} \log \Gamma(N_{ijk} + 1) \right],

    where :math:`r_i` is the cardinality of :math:`X_i`, :math:`q_i` is the number of parent configurations of
    :math:`\Pi_i`, :math:`N_{ijk}` is the count of :math:`X_i = k` in parent configuration :math:`j`, and
    :math:`N_{ij} = \sum_{k=1}^{r_i} N_{ijk}`.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable. Missing values should be set to `numpy.nan`.
    state_names : dict, optional
        Dictionary mapping each variable to its discrete states. If not specified, the unique values observed in the
        data are used.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.structure_score import K2
    >>> data = pd.DataFrame(
    ...     {"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]}
    ... )
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> score = K2(data)
    >>> round(score.score(model), 3)
    np.float64(-9.875)
    >>> round(score.local_score("B", ("A",)), 3)
    np.float64(-3.584)

    Raises
    ------
    ValueError
        If the data contains non-discrete variables, or if the model variables are not present in the data.

    References
    ----------
    .. [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009, Section 18.3.4-18.3.6.
    .. [2] AM Carvalho, Scoring functions for learning Bayesian networks,
        http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    _tags = {
        "name": "k2",
        "supported_datatype": "discrete",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
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
