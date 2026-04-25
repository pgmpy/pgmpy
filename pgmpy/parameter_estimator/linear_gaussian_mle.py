from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork

from .base import GaussianParameterEstimator


class LinearGaussianMLE(GaussianParameterEstimator):
    """
    Computes parameters for a LinearGaussianBayesianNetwork using Maximum Likelihood Estimation.

    For root nodes, the mean and variance are estimated directly from data. For non-root
    nodes, a linear regression is fit to estimate the regression coefficients and intercept;
    the residual standard deviation is used as the noise parameter.

    Parameters
    ----------
    std_estimator: {"unbiased", "mle"}, default="unbiased"
        Method used to estimate the noise standard deviation.
        - ``"unbiased"``: uses ``ddof = 1`` for root nodes and ``ddof = 1 + n_parents``
          for non-root nodes.
        - ``"mle"``: uses ``ddof = 0`` (biased, maximum likelihood estimate).

    Attributes
    ----------
    parameters_ : list of LinearGaussianCPD
        Learned Gaussian conditional probability distributions, one per
        variable in the model, ordered by `self._model.nodes()`. Populated by
        `fit`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.models import LinearGaussianBayesianNetwork
    >>> from pgmpy.parameter_estimator import LinearGaussianMLE
    >>> rng = np.random.default_rng(42)
    >>> data = pd.DataFrame(rng.normal(0, 1, (100, 3)), columns=["x1", "x2", "x3"])
    >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
    >>> estimator = LinearGaussianMLE()
    >>> estimator.fit(model, data).parameters_  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    [<LinearGaussianCPD: P(x1) = N(...) at 0x...>,
     <LinearGaussianCPD: P(x2 | x1) = N(...) at 0x...>,
     <LinearGaussianCPD: P(x3 | x2) = N(...) at 0x...>]
    """

    _tags = {
        "supports_latent_variables": False,
        "supports_weighted_data": False,
    }

    def __init__(self, std_estimator: str = "unbiased") -> None:
        self.std_estimator = std_estimator
        super().__init__()

    def fit(self, model: LinearGaussianBayesianNetwork, data, sample_weight=None) -> LinearGaussianMLE:
        """
        Estimate model parameters using Maximum Likelihood Estimation.

        Parameters
        ----------
        model: pgmpy.models.LinearGaussianBayesianNetwork
            The model structure for which to estimate CPDs.

        data: pandas.DataFrame
            DataFrame with column names identical to the variable names of the network.

        Returns
        -------
        self: LinearGaussianMLE
            Fitted estimator with learned CPDs stored in `parameters_`.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.parameter_estimator import LinearGaussianMLE
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(rng.normal(0, 1, (100, 3)), columns=["x1", "x2", "x3"])
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> estimator = LinearGaussianMLE()
        >>> estimator.fit(model, data).parameters_  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<LinearGaussianCPD: P(x1) = N(...) at 0x...>,
         <LinearGaussianCPD: P(x2 | x1) = N(...) at 0x...>,
         <LinearGaussianCPD: P(x3 | x2) = N(...) at 0x...>]
        """
        if self.std_estimator not in {"mle", "unbiased"}:
            raise ValueError(f"std_estimator must be one of {{'mle', 'unbiased'}}. Got: {self.std_estimator!r}")

        self._initialize_fit(model, data, sample_weight=sample_weight)

        cpds = []
        for node in self._model.nodes():
            parents = self._model.get_parents(node)

            if len(parents) == 0:
                ddof = 0 if self.std_estimator == "mle" else 1
                cpds.append(
                    LinearGaussianCPD(
                        variable=node,
                        beta=[self._data.loc[:, node].mean()],
                        std=self._data.loc[:, node].std(ddof=ddof),
                    )
                )
            else:
                lm = LinearRegression().fit(self._data.loc[:, parents], self._data.loc[:, node])
                residuals = self._data.loc[:, node] - lm.predict(self._data.loc[:, parents])
                ddof = 0 if self.std_estimator == "mle" else 1 + len(parents)
                cpds.append(
                    LinearGaussianCPD(
                        variable=node,
                        beta=np.append([lm.intercept_], lm.coef_),
                        std=residuals.std(ddof=ddof),
                        evidence=parents,
                    )
                )

        self.parameters_ = self._sort_parameters(cpds)
        return self
