from math import log

from pgmpy.structure_score.log_likelihood import LogLikelihood


class BIC(LogLikelihood):
    r"""
    BIC structure score for discrete Bayesian networks.

    BIC, also known as the MDL score, balances discrete log-likelihood against model complexity. The local score
    is computed as:

    .. math::
        \operatorname{BIC}(X_i, \Pi_i) = \ell(X_i, \Pi_i) - \frac{\log n}{2} q_i (r_i - 1),

    where :math:`\ell(X_i, \Pi_i)` is the local discrete log-likelihood, :math:`n` is the number of rows in `self.data`,
    :math:`q_i` is the number of parent configurations of :math:`\Pi_i`, and :math:`r_i` is the cardinality of
    :math:`X_i`.

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
    >>> from pgmpy.structure_score import BIC
    >>> data = pd.DataFrame(
    ...     {"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]}
    ... )
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> score = BIC(data)
    >>> round(score.score(model), 3)
    np.float64(-10.397)
    >>> round(score.local_score("B", ("A",)), 3)
    np.float64(-4.159)

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
        "name": "bic-d",
        "supported_datatype": "discrete",
        "default_for": "discrete",
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        sample_size = len(self.data)
        ll, num_parents_states, var_cardinality = self._log_likelihood(variable=variable, parents=parents)
        score = ll - 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)

        return score
