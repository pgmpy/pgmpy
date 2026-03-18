#!/usr/bin/env python
import numpy as np
import statsmodels.formula.api as smf

from pgmpy.causal_discovery.structure_score._base import BaseStructureScore


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
        "name": "ll-g",
        "supported_datatype": "continuous",
        "is_parametric": True,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

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
            glm_model = smf.glm(formula=f"{variable} ~ {' + '.join(parents)}", data=self.data).fit()

        return (glm_model.llf, glm_model.df_model)

    def local_score(self, variable, parents):
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
        "name": "bic-g",
        "supported_datatype": "continuous",
        "is_parametric": True,
        "is_default": True,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def local_score(self, variable, parents):
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
        "name": "aic-g",
        "supported_datatype": "continuous",
        "is_parametric": True,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def local_score(self, variable, parents):
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
