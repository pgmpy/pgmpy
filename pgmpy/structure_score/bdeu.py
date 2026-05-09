from math import lgamma

import numpy as np
from scipy.special import gammaln

from pgmpy.structure_score._base import BaseStructureScore
from pgmpy.utils import encode_columns, get_state_counts_array


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
    - :cite:p:`koller_friedman_2009`
    - :cite:p:`liao_2022`
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
        self._codes, self._cardinalities = encode_columns(self.data, self.state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        counts = get_state_counts_array(self._codes, self._cardinalities, variable, parents)
        num_parents_states = counts.shape[1]
        var_cardinality = self._cardinalities[variable]
        counts_size = num_parents_states * var_cardinality

        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts_size

        log_gamma_counts = np.zeros_like(counts)
        gammaln(counts + beta, out=log_gamma_counts)

        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + num_parents_states * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return score
