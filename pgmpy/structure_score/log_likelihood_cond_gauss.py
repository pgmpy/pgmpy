from collections.abc import Hashable

import numpy as np
import pandas as pd

from pgmpy.structure_score._base import BaseStructureScore

# Relative eigenvalue cutoff for singularity, as in `scipy.stats.multivariate_normal(allow_singular=True)`.
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
        "supported_datatype": "mixed",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None, max_cache_size=10000):
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)

    @staticmethod
    def _adjusted_cov(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """`df.cov()`, ridged or replaced by the identity when degenerate, and whether it was left unchanged."""
        n_cols = len(df.columns)
        identity = pd.DataFrame(np.eye(n_cols), index=df.columns, columns=df.columns)
        if (df.shape[0] == 1) or (df.shape[0] < n_cols):
            return identity, False

        df_cov = df.cov()
        eigenvalues = np.linalg.eigvalsh(df_cov.to_numpy())
        if eigenvalues[-1] <= 0:
            return identity, False
        if eigenvalues[0] <= _SINGULAR_RCOND * eigenvalues[-1]:
            return df_cov + 1e-6 * eigenvalues[-1] * np.eye(n_cols), False
        return df_cov, True

    @staticmethod
    def _gaussian_log_likelihood(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> float:
        """Total log density of the rows of `df` under the Gaussian fitted to `reference` (default `df`)."""
        n_rows, n_cols = df.shape
        if n_cols == 0:
            return 0.0

        fitted_on_df = reference is None
        reference = df if fitted_on_df else reference
        cov, is_sample_cov = LogLikelihoodCondGauss._adjusted_cov(reference)

        # A singular covariance uses the pseudo-determinant and pseudo-inverse.
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(cov, dtype=float))
        kept = eigenvalues > _SINGULAR_RCOND * np.max(np.abs(eigenvalues))
        rank = int(np.count_nonzero(kept))
        log_pdet = np.sum(np.log(eigenvalues[kept]))

        # Summed Mahalanobis term n_rows * tr(precision @ scatter); (n_rows - 1) * rank under `df`'s own fit.
        if fitted_on_df and is_sample_cov:
            mahalanobis = (n_rows - 1) * rank
        else:
            values = df.to_numpy(dtype=float)
            delta = values.mean(axis=0) - reference.mean(axis=0).to_numpy(dtype=float)
            scatter = np.cov(values, rowvar=False, ddof=0).reshape(n_cols, n_cols) + np.outer(delta, delta)
            basis = eigenvectors[:, kept]
            precision = (basis / eigenvalues[kept]) @ basis.T
            mahalanobis = n_rows * np.sum(precision * scatter)

        return -0.5 * (n_rows * (rank * np.log(2.0 * np.pi) + log_pdet) + mahalanobis)

    def _cat_parents_product(self, parents: tuple[Hashable, ...]) -> int:
        k = 1
        for pa in parents:
            if self.dtypes[pa] != "N":
                n_states = self.data[pa].nunique()
                if n_states > 1:
                    k *= self.data[pa].nunique()
        return k

    def _get_num_parameters(self, variable: Hashable, parents: tuple[Hashable, ...]) -> int:
        parent_dtypes = [self.dtypes[pa] for pa in parents]
        n_cont_parents = parent_dtypes.count("N")

        if self.dtypes[variable] == "N":
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
        parent_list = list(parents)
        df = self.data.loc[:, [variable] + parent_list]
        n_samples = df.shape[0]

        if self.dtypes[variable] == "N":
            c1 = variable
            c2 = [var for var in parents if self.dtypes[var] == "N"]
            d = list(set(parents) - set(c2))

            if len(d) == 0:
                return self._gaussian_log_likelihood(df) - self._gaussian_log_likelihood(df.loc[:, c2])

            log_like = 0
            for _, df_d in df.groupby(d, observed=True):
                log_like += self._gaussian_log_likelihood(df_d.loc[:, [c1] + c2])
                log_like -= self._gaussian_log_likelihood(df_d.loc[:, c2])
            return log_like

        d1 = variable
        c = [var for var in parents if self.dtypes[var] == "N"]
        d2 = list(set(parents) - set(c))

        d2_strata = dict(list(df.groupby(d2, observed=True))) if len(d2) > 0 else {(): df}

        log_like = 0
        for d_states, df_d1d2 in df.groupby([d1] + d2, observed=True):
            n_rows = df_d1d2.shape[0]
            df_d2 = d2_strata[d_states[1:]]
            log_like += self._gaussian_log_likelihood(df_d1d2.loc[:, c])
            log_like += n_rows * np.log(n_rows / n_samples)
            if len(c) > 0:
                log_like -= self._gaussian_log_likelihood(df_d1d2.loc[:, c], reference=df_d2.loc[:, c])
            if len(d2) > 0:
                log_like -= n_rows * np.log(df_d2[d1].count() / n_samples)
        return log_like

    def _local_score(self, variable: Hashable, parents: tuple[Hashable, ...]) -> float:
        ll = self._log_likelihood(variable=variable, parents=parents)
        return ll
