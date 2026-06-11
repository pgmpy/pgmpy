from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from scipy.special import gammaln

from pgmpy.structure_score._base import BaseStructureScore


class BGe(BaseStructureScore):
    r"""
    BGe (Bayesian Gaussian equivalent) structure score for Gaussian Bayesian networks.

    The BGe score is the marginal log-likelihood of the data under a Normal-Wishart
    prior on the joint Gaussian distribution. It is the score-equivalent Bayesian
    counterpart of :class:`BICGauss` and the continuous analogue of :class:`BDeu`.

    Hyperparameters of the Normal-Wishart prior:

    .. math::
        W \sim \text{Wishart}_n(\alpha_w, T_0), \qquad
        \mu \mid W \sim \mathcal{N}(\mu_0, (\alpha_\mu W)^{-1}).

    The prior scale matrix is hard-coded to the score-equivalent diagonal form

    .. math::
        T_0 = t \, I_n, \qquad
        t = \frac{\alpha_\mu (\alpha_w - n - 1)}{\alpha_\mu + 1},

    which ensures that Markov-equivalent DAGs receive identical scores.

    Per-family local score for node :math:`j` with parents :math:`\Pi_j`,
    :math:`l = |\Pi_j|`:

    .. math::
        \text{BGe}(X_j, \Pi_j) = c(l)
            + \tfrac{1}{2}(N + \alpha_w - n + l) \log |R_{\Pi_j, \Pi_j}|
            - \tfrac{1}{2}(N + \alpha_w - n + l + 1) \log |R_{\Pi_j \cup \{j\}, \Pi_j \cup \{j\}}|,

    where :math:`R = T_0 + S + \tfrac{N \alpha_\mu}{N + \alpha_\mu}
    (\bar{x} - \mu_0)(\bar{x} - \mu_0)^\top` is the posterior scale matrix
    (Kuipers, Moffa & Heckerman 2014 correction) and :math:`c(l)` is a
    family-size-dependent constant precomputed once.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.
    alpha_mu : float, default 1.0
        Imaginary sample size for the mean prior (bnlearn's `iss.mu`,
        BiDAG's `am`). Must be strictly positive.
    alpha_w : float, optional
        Degrees of freedom of the Wishart prior (bnlearn's `iss.w`,
        BiDAG's `aw`). Must satisfy ``alpha_w > n + 1``. Defaults to
        ``n + alpha_mu + 1``.
    mu_0 : {"zero", "mean"} or array-like, default "zero"
        Prior mean :math:`\mu_0`. ``"zero"`` uses zeros (BiDAG/DiBS
        convention). ``"mean"`` uses the sample column means (bnlearn
        convention; the prior-mean conflict term vanishes). An explicit
        length-``n`` array may also be supplied.
    state_names : dict, optional
        Accepted for API consistency but not typically used for Gaussian
        networks.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.structure_score import BGe
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": rng.normal(size=100),
    ...         "B": rng.normal(size=100),
    ...         "C": rng.normal(size=100),
    ...     }
    ... )
    >>> score = BGe(data)
    >>> isinstance(score.local_score("B", ("A", "C")), float)
    True

    Raises
    ------
    ValueError
        If ``alpha_mu`` is not positive, if ``alpha_w`` is not strictly
        greater than ``n + 1``, or if ``mu_0`` has an invalid value/shape.

    References
    ----------
    - Geiger, D. and Heckerman, D. (2002). Parameter Priors for Directed
      Acyclic Graphical Models and the Characterization of Several Probability
      Distributions. *Annals of Statistics* 30, 1412-1440.
    - Kuipers, J., Moffa, G. and Heckerman, D. (2014). Addendum on the
      scoring of Gaussian directed acyclic graphical models.
      *Annals of Statistics* 42(4), 1689-1691.
    """

    _tags = {
        "name": "bge",
        "supported_datatype": "continuous",
        "default_for": None,
        "is_parameteric": True,
    }

    def __init__(self, data, alpha_mu=1.0, alpha_w=None, mu_0="zero", state_names=None):
        super().__init__(data, state_names=state_names)

        self._np_data = self.data.to_numpy(dtype=float)
        self._col_index = {col: i for i, col in enumerate(self.data.columns)}
        N, n = self._np_data.shape

        if alpha_mu <= 0:
            raise ValueError(f"alpha_mu must be > 0; got {alpha_mu}.")
        if alpha_w is None:
            alpha_w = n + alpha_mu + 1
        if alpha_w <= n + 1:
            raise ValueError(f"alpha_w must be > n + 1 (= {n + 1}); got {alpha_w}.")

        if isinstance(mu_0, str):
            if mu_0 == "zero":
                mu_0_arr = np.zeros(n)
            elif mu_0 == "mean":
                mu_0_arr = self._np_data.mean(axis=0)
            else:
                raise ValueError(f"mu_0 must be 'zero', 'mean', or an array; got {mu_0!r}.")
        else:
            mu_0_arr = np.asarray(mu_0, dtype=float)
            if mu_0_arr.shape != (n,):
                raise ValueError(f"mu_0 must have shape ({n},); got {mu_0_arr.shape}.")

        self.alpha_mu = float(alpha_mu)
        self.alpha_w = float(alpha_w)
        self.mu_0 = mu_0_arr
        self._N = N
        self._n = n

        t = self.alpha_mu * (self.alpha_w - n - 1.0) / (self.alpha_mu + 1.0)
        self._t = t

        xbar = self._np_data.mean(axis=0)
        centered = self._np_data - xbar
        S = centered.T @ centered
        diff = (xbar - mu_0_arr).reshape(-1, 1)
        coef = (N * self.alpha_mu) / (N + self.alpha_mu)
        self._R = t * np.eye(n) + S + coef * (diff @ diff.T)

        ls = np.arange(n)
        self._scoreconstvec = (
            0.5 * (np.log(self.alpha_mu) - np.log(N + self.alpha_mu))
            - 0.5 * N * np.log(np.pi)
            + gammaln(0.5 * (N + self.alpha_w - n + ls + 1))
            - gammaln(0.5 * (self.alpha_w - n + ls + 1))
            + 0.5 * (self.alpha_w - n + 2 * ls + 1) * np.log(t)
        )

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        j = self._col_index[variable]
        l = len(parents)
        N, n = self._N, self._n
        awpNd2 = 0.5 * (N + self.alpha_w - n + l + 1)
        R = self._R
        Rjj = R[j, j]

        if l == 0:
            return float(self._scoreconstvec[0] - awpNd2 * np.log(Rjj))

        pa = [self._col_index[p] for p in parents]
        D = R[np.ix_(pa, pa)]
        B = R[j, pa]
        L, _ = cho_factor(D, lower=True)
        # cho_factor returns the full matrix with the upper triangle untouched;
        # we only read the diagonal and the lower triangle via solve_triangular.
        logdetD = 2.0 * np.sum(np.log(np.diag(L)))
        y = solve_triangular(L, B, lower=True)
        schur = Rjj - float(y @ y)
        return float(self._scoreconstvec[l] - 0.5 * logdetD - awpNd2 * np.log(schur))
