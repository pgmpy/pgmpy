from collections.abc import Hashable

import numpy as np

from pgmpy.structure_score._base import BaseStructureScore
from pgmpy.utils import is_discrete_type, residual_covariance, split_by_type

_SINGULAR_RCOND = 1e6 * np.finfo(float).eps


class LogLikelihoodCondGauss(BaseStructureScore):
    r"""
    Log-likelihood score for Bayesian networks with mixed discrete and continuous variables.

    This score is based on conditional Gaussian distributions [1] and supports local families with both discrete and
    continuous variables.

    For a continuous target :math:`C_1` with continuous parents :math:`C_2` and discrete parents :math:`D`, it computes

    .. math::
        \ell(C_1 \mid C_2, D) = \sum_{t=1}^{n} \log \frac{p(c_{1t}, c_{2t} \mid d_t)}{p(c_{2t} \mid d_t)}.

    For a discrete target :math:`D_1` with continuous parents :math:`C` and discrete parents :math:`D_2`, it computes

    .. math::
        \ell(D_1 \mid C, D_2) = \sum_{t=1}^{n} \log \frac{p(c_t \mid d_{1t}, d_{2t}) p(d_{1t}, d_{2t})} {p(c_t \mid
        d_{2t}) p(d_{2t})}.

    The Gaussian densities are estimated from the corresponding grouped samples.

    As in bnlearn, a family scores :math:`-\infty` when a continuous target has no residual variance left within some
    configuration of its discrete parents, for example when the configuration has no more rows than the regression has
    coefficients. A discrete target scores :math:`-\infty` when its continuous parents span fewer dimensions within some
    configuration of the target and its discrete parents than in the whole data.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where columns may be discrete or continuous variables.
    state_names : dict, optional
        Dictionary mapping discrete variable names to their possible states.
    max_cache_size : int or None, default=10000
        Maximum number of local scores to cache. If None, the cache is unlimited.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.structure_score import LogLikelihoodCondGauss
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": rng.normal(size=100),
    ...         "B": rng.integers(0, 2, size=100),
    ...         "C": rng.normal(size=100),
    ...     }
    ... )
    >>> score = LogLikelihoodCondGauss(data)
    >>> round(score.local_score("A", ("B", "C")), 3)
    np.float64(-137.319)

    Raises
    ------
    ValueError
        If the data or variable types are not suitable for conditional Gaussian modeling.

    References
    ----------
    - :footcite:t:`andrews_ramsey_cooper_2018`
    """

    _tags = {
        "name": "ll-cg",
        "data_types": ["discrete", "continuous", "mixed"],
        "default_for": None,
        "assumption:linearity": True,
        "assumption:additive_noise": True,
        "assumption:gaussian_noise": True,
    }

    def __init__(self, data, state_names=None, max_cache_size=10000):
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)

    @staticmethod
    def _gaussian_fit(values: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray]:
        """Rank, log pseudo-determinant and pseudo-inverse of the sample covariance of `values`, and their mean."""
        n_rows, n_cols = values.shape
        if n_rows < 2 or n_cols == 0:
            return 0, 0.0, np.zeros((n_cols, n_cols)), values.mean(axis=0)

        eigenvalues, eigenvectors = np.linalg.eigh(np.atleast_2d(np.cov(values, rowvar=False)))
        kept = eigenvalues > _SINGULAR_RCOND * max(eigenvalues[-1], 1.0)
        basis = eigenvectors[:, kept]
        precision = (basis / eigenvalues[kept]) @ basis.T
        return int(np.count_nonzero(kept)), float(np.sum(np.log(eigenvalues[kept]))), precision, values.mean(axis=0)

    def _cat_parents_product(self, parents: tuple[Hashable, ...]) -> int:
        k = 1
        for pa in parents:
            if is_discrete_type(self.dtypes[pa]):
                n_states = self.data[pa].nunique()
                if n_states > 1:
                    k *= self.data[pa].nunique()
        return k

    def _get_num_parameters(self, variable: Hashable, parents: tuple[Hashable, ...]) -> int:
        n_cont_parents = len(split_by_type(parents, self.dtypes)[0])

        if not is_discrete_type(self.dtypes[variable]):
            k = self._cat_parents_product(parents=parents) * (n_cont_parents + 2)
        else:
            if n_cont_parents == 0:
                k = self._cat_parents_product(parents=parents) * (self.data[variable].nunique() - 1)
            else:
                k = (
                    self._cat_parents_product(parents=parents)
                    * (self.data[variable].nunique() - 1)
                    * (n_cont_parents + 2)
                )

        return k

    def _log_likelihood(self, variable: Hashable, parents: tuple[Hashable, ...]) -> float:
        df = self.data.loc[:, [variable, *parents]]
        n_samples = df.shape[0]
        continuous = split_by_type([variable, *parents], self.dtypes)[0]
        discrete = split_by_type(parents, self.dtypes)[1]

        # Step 1: Standardize the continuous variables; an exactly constant one is only centered.
        values = df.loc[:, continuous].to_numpy(dtype=float)
        scale = values.std(axis=0)
        scale[np.ptp(values, axis=0) == 0] = 1.0
        values = (values - values.mean(axis=0)) / scale

        if not is_discrete_type(self.dtypes[variable]):
            # Step 2: Add the Gaussian log-likelihood of the target's regression within each discrete configuration.
            strata = list(df.groupby(discrete, observed=True).indices.values()) if discrete else [np.arange(n_samples)]
            log_like = 0.0
            for rows in strata:
                n_rows = len(rows)
                stratum_cov = np.atleast_2d(np.cov(values[rows], rowvar=False, ddof=0))
                residual_var = residual_covariance(
                    stratum_cov, [0], list(range(1, len(continuous))), values=values[rows]
                )[0, 0]
                if residual_var <= np.finfo(float).eps:
                    return -np.inf
                log_like -= 0.5 * (n_rows * np.log(2.0 * np.pi * residual_var * n_rows / (n_rows - 1)) + n_rows - 1)

            # Step 3: Undo the standardization of the target.
            return log_like - sum(len(rows) for rows in strata) * np.log(scale[0])

        # Step 2: Fit the continuous parents within each configuration of the discrete parents.
        d2_ids = df.groupby(discrete, observed=True).ngroup().to_numpy() if discrete else np.zeros(n_samples, dtype=int)
        known_target = df[variable].notna().to_numpy()
        d2_counts, d2_fits = {}, {}
        for d2_id in np.unique(d2_ids[d2_ids >= 0]):
            d2_rows = d2_ids == d2_id
            d2_counts[d2_id] = np.count_nonzero(known_target & d2_rows) if discrete else n_samples
            d2_fits[d2_id] = self._gaussian_fit(values[d2_rows])
        global_rank = self._gaussian_fit(values)[0]

        # Step 3: Add log p(d1, d2) / p(d2) and the log density ratio of the continuous parents given (d1, d2) and d2.
        log_like = 0.0
        for rows in df.groupby([variable, *discrete], observed=True).indices.values():
            n_rows = len(rows)
            d2_id = d2_ids[rows[0]]
            rank, log_pdet, _, _ = self._gaussian_fit(values[rows])
            d2_rank, d2_log_pdet, d2_precision, d2_mean = d2_fits[d2_id]
            if rank != global_rank or d2_rank != global_rank:
                return -np.inf

            centered = values[rows] - d2_mean
            mahalanobis = np.sum((centered @ d2_precision) * centered)
            log_like += n_rows * (np.log(n_rows / n_samples) - np.log(d2_counts[d2_id] / n_samples))
            log_like -= 0.5 * (n_rows * (log_pdet - d2_log_pdet) + (n_rows - 1) * rank - mahalanobis)
        return log_like

    def _local_score(self, variable: Hashable, parents: tuple[Hashable, ...]) -> float:
        ll = self._log_likelihood(variable=variable, parents=parents)
        return ll
