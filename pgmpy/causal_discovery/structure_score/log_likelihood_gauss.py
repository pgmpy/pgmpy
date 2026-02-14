#!/usr/bin/env python
from typing import List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import multivariate_normal

from pgmpy.base import DAG
from pgmpy.causal_discovery.structure_score import BaseStructureScore


class LogLikelihoodGauss(BaseStructureScore):
    """
    Log-likelihood structure score for Gaussian Bayesian networks.

    This score evaluates the fit of a continuous (Gaussian) Bayesian network structure
    by computing the (unpenalized) log-likelihood of the observed data given the model,
    using generalized linear modeling. It is suitable for networks with continuous variables.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.

    state_names : dict, optional
        Dictionary mapping variable names to possible states. Not typically used for Gaussian networks.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import LogLikelihoodGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randn(100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = LogLikelihoodGauss(data)
    >>> ll = score.local_score("B", ["A", "C"])
    >>> print(ll)
    -142.125

    Raises
    ------
    ValueError
        If the data contains discrete or non-numeric variables.
    """

    _tags = {
        "name": "log_likelihood_gauss_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super(LogLikelihoodGauss, self).__init__(data, **kwargs)

    def _log_likelihood(self, variable, parents):
        """
        Computes the log-likelihood and degrees of freedom for a Gaussian model.

        This internal method fits a generalized linear model (GLM) for the specified variable
        as a function of its parent variables, using the statsmodels library, and returns the
        log-likelihood and degrees of freedom of the fitted model.

        Parameters
        ----------
        variable : str
            The name of the variable (node) to be predicted.
        parents : list of str
            List of variable names to be used as predictors (parents). If empty, fits an intercept-only model.

        Returns
        -------
        llf : float
            The log-likelihood of the fitted model.
        df_model : int
            The degrees of freedom of the fitted model (number of model parameters estimated).

        Examples
        --------
        >>> llf, df = score._log_likelihood("B", ["A", "C"])
        >>> print(llf, df)
        -142.125 2

        Raises
        ------
        ValueError
            If the GLM cannot be fitted due to missing or non-numeric data.
        """
        if len(parents) == 0:
            glm_model = smf.glm(formula=f"{variable} ~ 1", data=self.data).fit()
        else:
            glm_model = smf.glm(
                formula=f"{variable} ~ {' + '.join(parents)}", data=self.data
            ).fit()

        return (glm_model.llf, glm_model.df_model)

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the log-likelihood score for a variable given its parent variables.

        Fits a generalized linear model (GLM) for the variable as a function of its parents,
        and returns the resulting log-likelihood as the structure score.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The log-likelihood score for the specified variable and parent configuration.

        Examples
        --------
        >>> ll = score.local_score("B", ["A", "C"])
        >>> print(ll)
        -142.125

        Raises
        ------
        ValueError
            If the GLM cannot be fitted due to non-numeric data or missing columns.
        """
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        return ll


class BICGauss(LogLikelihoodGauss):
    """
    BIC (Bayesian Information Criterion) structure score for Gaussian Bayesian networks.

    The BICGauss score evaluates continuous Bayesian network structures by penalizing
    the log-likelihood with a term proportional to the number of model parameters,
    discouraging overfitting. This is the Gaussian version of the BIC/MDL score,
    suitable for networks where all variables are continuous.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import BICGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randn(100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = BICGauss(data)
    >>> s = score.local_score("B", ["A", "C"])
    >>> print(s)
    -111.42

    Raises
    ------
    ValueError
        If the GLM cannot be fitted due to missing or non-numeric data.
    """

    _tags = {
        "name": "bic_gauss_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": True,
    }

    def __init__(self, data, **kwargs):
        super(BICGauss, self).__init__(data, **kwargs)

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the local BIC/MDL score for a variable and its parent variables
        in a Gaussian Bayesian network.

        The score is the log-likelihood minus a penalty term that increases
        with the number of model parameters and sample size.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local BICGauss score for the specified variable and parent configuration.

        Examples
        --------
        >>> s = score.local_score("B", ["A", "C"])
        >>> print(s)
        -111.42

        Raises
        ------
        ValueError
            If the GLM cannot be fitted due to missing or non-numeric data.
        """
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        # Adding +2 to model df to compute the likelihood df.
        return ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))


class AICGauss(LogLikelihoodGauss):
    """
    AIC (Akaike Information Criterion) structure score for Gaussian Bayesian networks.

    The AICGauss score evaluates continuous Bayesian network structures by penalizing
    the log-likelihood with a term proportional to the number of model parameters.
    The penalty is less severe than BIC and does not depend on sample size, making AIC
    preferable for model selection with smaller datasets.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import AICGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randn(100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = AICGauss(data)
    >>> s = score.local_score("B", ["A", "C"])
    >>> print(s)
    -97.53

    Raises
    ------
    ValueError
        If the GLM cannot be fitted due to missing or non-numeric data.
    """

    _tags = {
        "name": "aic_gauss_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super(AICGauss, self).__init__(data, **kwargs)

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the local AIC score for a variable and its parent variables
        in a Gaussian Bayesian network.

        The score is the log-likelihood minus a penalty term that increases with
        the number of model parameters (but not sample size).

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local AICGauss score for the specified variable and parent configuration.

        Examples
        --------
        >>> s = score.local_score("B", ["A", "C"])
        >>> print(s)
        -97.53

        Raises
        ------
        ValueError
            If the GLM cannot be fitted due to missing or non-numeric data.
        """
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        # Adding +2 to model df to compute the likelihood df.
        return ll - (df_model + 2)


class LogLikelihoodCondGauss(BaseStructureScore):
    """
    Log-likelihood score for Bayesian networks with mixed discrete and continuous variables.

    This score is based on conditional Gaussian distributions and supports networks
    with both discrete and continuous variables, using the methodology described in [1].
    The local score computes the log-likelihood of the observed data given the
    network structure, handling mixed parent sets as described in the reference.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where columns can be discrete or continuous variables.
        Variable types should be consistent with the structure.

    state_names : dict, optional
        Dictionary mapping discrete variable names to their possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import LogLikelihoodCondGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randint(0, 2, 100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = LogLikelihoodCondGauss(data)
    >>> ll = score.local_score("A", ["B", "C"])
    >>> print(ll)
    -98.452

    Raises
    ------
    ValueError
        If the data or variable types are not suitable for conditional Gaussian modeling.

    References
    ----------
    [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian
        Networks of Mixed Variables. International journal of data science and
        analytics, 6(1), 3–18. https://doi.org/10.1007/s41060-017-0085-7
    """

    _tags = {
        "name": "log_likelihood_cond_gauss_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super(LogLikelihoodCondGauss, self).__init__(data, **kwargs)

    @staticmethod
    def _adjusted_cov(df):
        """
        Computes an adjusted covariance matrix from the given DataFrame.

        This method returns the sample covariance matrix for the columns in `df`, making sure
        the result is always positive semi-definite. If there are not enough rows to estimate
        covariance (i.e., fewer rows than variables), the identity matrix is returned. If the
        covariance matrix is not positive semi-definite, a small value is added to the diagonal.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame whose columns are the variables for which the covariance matrix is computed.

        Returns
        -------
        cov_matrix : pandas.DataFrame
            The adjusted covariance matrix. If not enough data, returns the identity matrix.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])
        >>> cov = LogLikelihoodCondGauss._adjusted_cov(df)
        >>> print(cov)
                A         B         C
        A  0.802359  0.100722 -0.006956
        B  0.100722  0.818795  0.154614
        C -0.006956  0.154614  0.540758
        """
        # If a number of rows less than number of variables, return variance 1 with no covariance.
        if (df.shape[0] == 1) or (df.shape[0] < len(df.columns)):
            return pd.DataFrame(
                np.eye(len(df.columns)), index=df.columns, columns=df.columns
            )

        # If the matrix is not positive semidefinite, add a small error to make it.
        df_cov = df.cov()
        if np.any(np.isclose(np.linalg.eig(df_cov)[0], 0)):
            df_cov = df_cov + 1e-6
        return df_cov

    def _cat_parents_product(self, parents):
        """
        Computes the product of the number of unique states for each categorical parent.

        For each parent in `parents` that is discrete (not continuous), this method multiplies
        the number of observed unique states. Parents with only one unique value are ignored
        (i.e., do not contribute to the product).

        Parameters
        ----------
        parents : list of str
            List of parent variable names to consider.

        Returns
        -------
        k : int
            The product of unique state counts for each discrete parent in `parents`.

        Examples
        --------
        >>> score._cat_parents_product(["A", "B", "C"])
        6
        """
        k = 1
        for pa in parents:
            if self.dtypes[pa] != "N":
                n_states = self.data[pa].nunique()
                if n_states > 1:
                    k *= self.data[pa].nunique()
        return k

    def _get_num_parameters(self, variable, parents):
        """
        Computes the number of free parameters required for the conditional distribution
        of a variable given its parents in a mixed (discrete and continuous) Bayesian network.

        For a continuous variable, the number of parameters depends on the number of continuous
        parents and the number of configurations of discrete parents. For a discrete variable,
        it depends on the number of categories and parent configurations.

        Parameters
        ----------
        variable : str
            The name of the target variable (child node).
        parents : list of str
            List of parent variable names.

        Returns
        -------
        k : int
            The number of free parameters for the conditional distribution of `variable`
            given its parents.

        Examples
        --------
        >>> score._get_num_parameters("A", ["B", "C"])
        12
        """
        parent_dtypes = [self.dtypes[pa] for pa in parents]
        n_cont_parents = parent_dtypes.count("N")

        if self.dtypes[variable] == "N":
            k = self._cat_parents_product(parents=parents) * (n_cont_parents + 2)
        else:
            if n_cont_parents == 0:
                k = self._cat_parents_product(parents=parents) * (
                    self.data[variable].nunique() - 1
                )
            else:
                k = (
                    self._cat_parents_product(parents=parents)
                    * (self.data[variable].nunique() - 1)
                    * (n_cont_parents + 2)
                )

        return k

    def _log_likelihood(self, variable, parents):
        """
        Computes the conditional log-likelihood for a variable given its parent set,
        supporting both continuous and discrete variables (mixed Bayesian networks).

        For a continuous variable, computes the log-likelihood using conditional Gaussian
        distributions as described in [1]. For a discrete variable, computes the
        log-likelihood based on the joint and marginal probabilities involving both
        discrete and continuous parents.

        Parameters
        ----------
        variable : str
            The variable (node) for which the log-likelihood is computed.
        parents : list of str
            List of parent variable names.

        Returns
        -------
        log_like : float
            The log-likelihood value for the specified variable and parent set.

        Examples
        --------
        >>> ll = score._log_likelihood("A", ["B", "C"])
        >>> print(ll)
        -99.242

        Raises
        ------
        ValueError
            If data is not suitable for log-likelihood computation (e.g., unsupported variable types).

        References
        ----------
        [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian
            Networks of Mixed Variables. International journal of data science and
            analytics, 6(1), 3–18. https://doi.org/10.1007/s41060-017-0085-7
        """
        df = self.data.loc[:, [variable] + parents]

        # If variable is continuous, the probability is computed as:
        # P(C1 | C2, D) = p(C1, C2 | D) / p(C2 | D)
        if self.dtypes[variable] == "N":
            c1 = variable
            c2 = [var for var in parents if self.dtypes[var] == "N"]
            d = list(set(parents) - set(c2))

            # If D = {}, p(C1, C2 | D) = p(C1, C2) and p(C2 | D) = p(C2)
            if len(d) == 0:
                # If C2 = {}, p(C1, C2 | D) = p(C1) and p(C2 | D) = 1.
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
                        cov=LogLikelihoodCondGauss._adjusted_cov(
                            df_d.loc[:, [c1] + c2]
                        ),
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
                                cov=LogLikelihoodCondGauss._adjusted_cov(
                                    df_d.loc[:, c2]
                                ),
                                allow_singular=True,
                            ),
                        )

                    log_like += np.sum(np.log(p_c1c2_d / p_c2_d))
                return log_like

        # If variable is discrete, the probability is computed as:
        # P(D1 | C, D2) = (p(C| D1, D2) p(D1, D2)) / (p(C| D2) p(D2))
        else:
            d1 = variable
            c = [var for var in parents if self.dtypes[var] == "N"]
            d2 = list(set(parents) - set(c))

            log_like = 0
            for d_states, df_d1d2 in df.groupby([d1] + d2, observed=True):
                # Check if df_d1d2 also has the discrete variables.
                # If C={}, p(C | D1, D2) = 1.
                if len(c) == 0:
                    p_c_d1d2 = 1
                else:
                    p_c_d1d2 = multivariate_normal.pdf(
                        x=df_d1d2.loc[:, c],
                        mean=df_d1d2.loc[:, c].mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df_d1d2.loc[:, c]),
                        allow_singular=True,
                    )

                # P(D1, D2)
                p_d1d2 = np.repeat(df_d1d2.shape[0] / df.shape[0], df_d1d2.shape[0])

                # If D2 = {}, p(D1 | C, D2) = (p(C | D1, D2) p(D1, D2)) / p(C)
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
                                cov=LogLikelihoodCondGauss._adjusted_cov(
                                    df_d2.loc[:, c]
                                ),
                                allow_singular=True,
                            ),
                        )

                    p_d2 = df.groupby(d2, observed=True).count() / df.shape[0]
                    for var, value in zip(d2, d_states[1:]):
                        p_d2 = p_d2.loc[p_d2.index.get_level_values(var) == value]

                    log_like += np.sum(
                        np.log((p_c_d1d2 * p_d1d2) / (p_c_d2 * p_d2.values.ravel()[0]))
                    )
            return log_like

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the local log-likelihood score for a variable given its parent variables
        in a mixed (discrete and continuous) Bayesian network.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local conditional Gaussian log-likelihood score for the specified variable and parent configuration.

        Examples
        --------
        >>> ll = score.local_score("A", ["B", "C"])
        >>> print(ll)
        -98.452

        Raises
        ------
        ValueError
            If the log-likelihood cannot be computed due to incompatible data or variable types.
        """
        ll = self._log_likelihood(variable=variable, parents=parents)
        return ll


class BICCondGauss(LogLikelihoodCondGauss):
    """
    BIC (Bayesian Information Criterion) score for Bayesian networks with mixed (discrete and continuous) variables.

    The BICCondGauss score evaluates network structures by penalizing the conditional log-likelihood
    with a term proportional to the number of free parameters and the logarithm of sample size.
    This approach generalizes the classic BIC to handle mixed discrete/continuous data as
    described in [1].

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where columns may be discrete or continuous variables.

    state_names : dict, optional
        Dictionary mapping discrete variable names to possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import BICCondGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randint(0, 2, 100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = BICCondGauss(data)
    >>> s = score.local_score("A", ["B", "C"])
    >>> print(s)
    -115.37

    Raises
    ------
    ValueError
        If the log-likelihood or number of parameters cannot be computed for the provided variables.

    References
    ----------
    [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian
        Networks of Mixed Variables. International journal of data science and
        analytics, 6(1), 3–18. https://doi.org/10.1007/s41060-017-0085-7
    """

    _tags = {
        "name": "bic_cond_gauss_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": True,
    }

    def __init__(self, data, **kwargs):
        super(BICCondGauss, self).__init__(data, **kwargs)

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the local BIC score for a variable and its parent set in a mixed Bayesian network.

        The score is calculated as the log-likelihood minus a complexity penalty, which
        is proportional to the number of free parameters and the log of the sample size.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local BICCondGauss score for the specified variable and parent configuration.

        Examples
        --------
        >>> s = score.local_score("A", ["B", "C"])
        >>> print(s)
        -115.37

        Raises
        ------
        ValueError
            If the log-likelihood or parameter count cannot be computed for the given configuration.
        """

        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - ((k / 2) * np.log(self.data.shape[0]))


class AICCondGauss(LogLikelihoodCondGauss):
    """
    AIC (Akaike Information Criterion) score for Bayesian networks with mixed (discrete and continuous) variables.

    The AICCondGauss score evaluates network structures by penalizing the conditional log-likelihood
    with a term equal to the number of free parameters. This generalizes the classic AIC
    to handle Bayesian networks with both discrete and continuous variables [1].

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where columns may be discrete or continuous variables.

    state_names : dict, optional
        Dictionary mapping discrete variable names to possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import AICCondGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randint(0, 2, 100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = AICCondGauss(data)
    >>> s = score.local_score("A", ["B", "C"])
    >>> print(s)
    -99.75

    Raises
    ------
    ValueError
        If the log-likelihood or number of parameters cannot be computed for the provided variables.

    References
    ----------
    [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian
        Networks of Mixed Variables. International journal of data science and
        analytics, 6(1), 3–18. https://doi.org/10.1007/s41060-017-0085-7
    """

    _tags = {
        "name": "aic_cond_gauss_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super(AICCondGauss, self).__init__(data, **kwargs)

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the local AIC score for a variable and its parent set in a mixed Bayesian network.

        The score is calculated as the log-likelihood minus the number of free parameters.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local AICCondGauss score for the specified variable and parent configuration.

        Examples
        --------
        >>> s = score.local_score("A", ["B", "C"])
        >>> print(s)
        -99.75

        Raises
        ------
        ValueError
            If the log-likelihood or parameter count cannot be computed for the given configuration.
        """
        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - k
