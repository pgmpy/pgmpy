import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from pgmpy.structure_score._base import BaseStructureScore


class LogLikelihoodCondGauss(BaseStructureScore):
    """Conditional-Gaussian log-likelihood structure score."""

    _tags = {
        "name": "ll-cg",
        "supported_datatype": "mixed",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    @staticmethod
    def _adjusted_cov(df: pd.DataFrame) -> pd.DataFrame:
        if (df.shape[0] == 1) or (df.shape[0] < len(df.columns)):
            return pd.DataFrame(np.eye(len(df.columns)), index=df.columns, columns=df.columns)

        df_cov = df.cov()
        if np.any(np.isclose(np.linalg.eig(df_cov)[0], 0)):
            df_cov = df_cov + 1e-6
        return df_cov

    def _cat_parents_product(self, parents: tuple[str, ...]) -> int:
        k = 1
        for pa in parents:
            if self.dtypes[pa] != "N":
                n_states = self.data[pa].nunique()
                if n_states > 1:
                    k *= self.data[pa].nunique()
        return k

    def _get_num_parameters(self, variable: str, parents: tuple[str, ...]) -> int:
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

    def _log_likelihood(self, variable: str, parents: tuple[str, ...]) -> float:
        parent_list = list(parents)
        df = self.data.loc[:, [variable] + parent_list]

        if self.dtypes[variable] == "N":
            c1 = variable
            c2 = [var for var in parents if self.dtypes[var] == "N"]
            d = list(set(parents) - set(c2))

            if len(d) == 0:
                if len(c2) == 0:
                    p_c1c2_d = multivariate_normal.pdf(
                        x=df,
                        mean=df.mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df),
                        allow_singular=True,
                    )
                    return np.sum(np.log(p_c1c2_d))
                else:
                    p_c1c2_d = multivariate_normal.pdf(
                        x=df,
                        mean=df.mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df),
                        allow_singular=True,
                    )
                    df_c2 = df.loc[:, c2]
                    p_c2_d = np.maximum(
                        1e-8,
                        multivariate_normal.pdf(
                            x=df_c2,
                            mean=df_c2.mean(axis=0),
                            cov=LogLikelihoodCondGauss._adjusted_cov(df_c2),
                            allow_singular=True,
                        ),
                    )

                    return np.sum(np.log(p_c1c2_d / p_c2_d))
            else:
                log_like = 0
                for d_states, df_d in df.groupby(d, observed=True):
                    p_c1c2_d = multivariate_normal.pdf(
                        x=df_d.loc[:, [c1] + c2],
                        mean=df_d.loc[:, [c1] + c2].mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df_d.loc[:, [c1] + c2]),
                        allow_singular=True,
                    )
                    if len(c2) == 0:
                        p_c2_d = 1
                    else:
                        p_c2_d = np.maximum(
                            1e-8,
                            multivariate_normal.pdf(
                                x=df_d.loc[:, c2],
                                mean=df_d.loc[:, c2].mean(axis=0),
                                cov=LogLikelihoodCondGauss._adjusted_cov(df_d.loc[:, c2]),
                                allow_singular=True,
                            ),
                        )

                    log_like += np.sum(np.log(p_c1c2_d / p_c2_d))
                return log_like

        else:
            d1 = variable
            c = [var for var in parents if self.dtypes[var] == "N"]
            d2 = list(set(parents) - set(c))

            log_like = 0
            for d_states, df_d1d2 in df.groupby([d1] + d2, observed=True):
                if len(c) == 0:
                    p_c_d1d2 = 1
                else:
                    p_c_d1d2 = multivariate_normal.pdf(
                        x=df_d1d2.loc[:, c],
                        mean=df_d1d2.loc[:, c].mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df_d1d2.loc[:, c]),
                        allow_singular=True,
                    )

                p_d1d2 = np.repeat(df_d1d2.shape[0] / df.shape[0], df_d1d2.shape[0])

                if len(d2) == 0:
                    if len(c) == 0:
                        p_c_d2 = 1
                    else:
                        p_c_d2 = np.maximum(
                            1e-8,
                            multivariate_normal.pdf(
                                x=df_d1d2.loc[:, c],
                                mean=df.loc[:, c].mean(axis=0),
                                cov=LogLikelihoodCondGauss._adjusted_cov(df.loc[:, c]),
                                allow_singular=True,
                            ),
                        )

                    log_like += np.sum(np.log(p_c_d1d2 * p_d1d2 / p_c_d2))
                else:
                    if len(c) == 0:
                        p_c_d2 = 1
                    else:
                        df_d2 = df
                        for var, state in zip(d2, d_states[1:]):
                            df_d2 = df_d2.loc[df_d2[var] == state]

                        p_c_d2 = np.maximum(
                            1e-8,
                            multivariate_normal.pdf(
                                x=df_d1d2.loc[:, c],
                                mean=df_d2.loc[:, c].mean(axis=0),
                                cov=LogLikelihoodCondGauss._adjusted_cov(df_d2.loc[:, c]),
                                allow_singular=True,
                            ),
                        )

                    p_d2 = df.groupby(d2, observed=True).count() / df.shape[0]
                    for var, value in zip(d2, d_states[1:]):
                        p_d2 = p_d2.loc[p_d2.index.get_level_values(var) == value]

                    log_like += np.sum(np.log((p_c_d1d2 * p_d1d2) / (p_c_d2 * p_d2.values.ravel()[0])))
            return log_like

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """Compute the local conditional-Gaussian log-likelihood score."""
        ll = self._log_likelihood(variable=variable, parents=parents)
        return ll
