from __future__ import annotations

import numpy as np

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.utils import _check_no_missing_values, covariance_sufficient_stats

from .base import GaussianParameterEstimator


class LinearGaussianMLE(GaussianParameterEstimator):
    """
    Computes parameters for a LinearGaussianBayesianNetwork using Maximum Likelihood Estimation.

    For root nodes, the mean and variance are estimated directly from data. For non-root
    nodes, a linear regression is fit to estimate the regression coefficients and intercept;
    the residual standard deviation is used as the noise parameter.

    Parameters
    ----------
    std_estimator : {"unbiased", "mle"}, default="unbiased"
        Method used to estimate the noise standard deviation.
        ``"unbiased"`` uses ``ddof = 1`` for root nodes and
        ``ddof = 1 + n_parents`` for non-root nodes. ``"mle"`` uses
        ``ddof = 0`` (biased, maximum likelihood estimate).

    Attributes
    ----------
    parameters_ : list of LinearGaussianCPD
        Learned Gaussian conditional probability distributions, one per
        variable in the model, ordered by ``self._model.nodes()``. Populated by
        ``fit``.

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

        nodes = list(self._model.nodes())
        cov, means, col_index, missing = covariance_sufficient_stats(self._data.loc[:, nodes])
        _check_no_missing_values(nodes, missing, "Maximum likelihood estimation of a linear Gaussian network")
        n_samples = self._data.shape[0]

        cpds = []
        for node in nodes:
            parents = self._model.get_parents(node)
            node_col = col_index[node]

            if len(parents) == 0:
                ddof = 0 if self.std_estimator == "mle" else 1
                cpds.append(
                    LinearGaussianCPD(
                        variable=node,
                        beta=[means[node_col]],
                        std=np.sqrt(n_samples * cov[node_col, node_col] / (n_samples - ddof)),
                    )
                )
            else:
                parent_cols = [col_index[parent] for parent in parents]
                coef = np.linalg.pinv(cov[np.ix_(parent_cols, parent_cols)]) @ cov[parent_cols, node_col]
                intercept = means[node_col] - coef @ means[parent_cols]
                residual_var = cov[node_col, node_col] - cov[node_col, parent_cols] @ coef
                if residual_var <= np.sqrt(np.finfo(float).eps) * cov[node_col, node_col]:
                    raise ValueError(
                        f"Cannot estimate the CPD of {node!r}: its parents {parents} determine it to within "
                        f"floating point precision, leaving no residual variance to estimate. Drop one of the "
                        f"redundant variables."
                    )

                ddof = 0 if self.std_estimator == "mle" else 1 + len(parents)
                cpds.append(
                    LinearGaussianCPD(
                        variable=node,
                        beta=np.append([intercept], coef),
                        std=np.sqrt(n_samples * residual_var / (n_samples - ddof)),
                        evidence=parents,
                    )
                )

        self.parameters_ = self._sort_parameters(cpds)
        return self
