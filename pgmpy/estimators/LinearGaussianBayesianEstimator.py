from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pgmpy.estimators import ParameterEstimator
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


class LinearGaussianBayesianEstimator(ParameterEstimator):
    """
    Bayesian parameter estimator for LinearGaussianBayesianNetwork using
    Normal-Inverse-Gamma conjugate priors.

    For each node, the model assumes:
        y = X @ beta + epsilon,  epsilon ~ N(0, sigma^2)

    with conjugate priors:
        beta | sigma^2 ~ N(B0, sigma^2 * V0)
        sigma^2        ~ InvGamma(alpha_0, beta_0)

    This yields closed-form posterior updates:
        Vn    = inv(inv(V0) + X^T X)
        Bn    = Vn @ (inv(V0) @ B0 + X^T y)
        alpha_n = alpha_0 + n/2
        beta_n  = beta_0 + 0.5 * (y^T y + B0^T inv(V0) B0 - Bn^T inv(Vn) Bn)

    The posterior mean of variance is beta_n / (alpha_n - 1).

    Parameters
    ----------
    model: LinearGaussianBayesianNetwork
        The model whose parameters are to be estimated.

    data: pd.DataFrame
        Continuous-valued data containing all model variables.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.models import LinearGaussianBayesianNetwork
    >>> from pgmpy.estimators import LinearGaussianBayesianEstimator
    >>> np.random.seed(42)
    >>> data = pd.DataFrame({"X": np.random.randn(100), "Y": np.random.randn(100)})
    >>> model = LinearGaussianBayesianNetwork([("X", "Y")])
    >>> estimator = LinearGaussianBayesianEstimator(model, data)
    >>> estimator.get_parameters()  # doctest: +ELLIPSIS
    [<LinearGaussianCPD: P(X) = N(-0.104; 0.906) at 0x...,
    <LinearGaussianCPD: P(Y | X) = N(-0.143*X + 0.007; 0.941) at 0x...]  # noqa: E501

    """

    def __init__(
        self, model: LinearGaussianBayesianNetwork, data: pd.DataFrame, **kwargs
    ):
        if not isinstance(model, LinearGaussianBayesianNetwork):
            raise NotImplementedError(
                "Only implemented for LinearGaussianBayesianNetwork"
            )
        super(LinearGaussianBayesianEstimator, self).__init__(model, data, **kwargs)

    def get_parameters(
        self,
        B0: Optional[Union[np.ndarray, Dict]] = None,
        V0: Optional[Union[np.ndarray, Dict]] = None,
        alpha_0: Union[float, Dict] = 2.0,
        beta_0: Union[float, Dict] = 1.0,
        n_jobs: int = 1,
    ) -> List[LinearGaussianCPD]:
        """
        Estimates the LinearGaussianCPD for every node in the model.

        Parameters
        ----------
        B0: np.ndarray or dict or None
            Prior mean vector for beta. Shape must be (k,) where
            k = 1 + number_of_parents for each node.
            - If np.ndarray, the same prior is used for all nodes.
            - If dict, keys are node names and values are per-node arrays.
            - If None, defaults to np.zeros(k) for each node.

        V0: np.ndarray or dict or None
            Prior covariance matrix for beta. Shape must be (k, k).
            - If np.ndarray, the same prior is used for all nodes.
            - If dict, keys are node names and values are per-node matrices.
            - If None, defaults to np.eye(k) * 10 for each node.

        alpha_0: float or dict
            Shape parameter of the Inverse-Gamma prior on variance.
            Must be > 1 for a finite posterior mean. Default is 2.0.
            - If dict, keys are node names and values are per-node floats.

        beta_0: float or dict
            Scale parameter of the Inverse-Gamma prior on variance.
            Default is 1.0.
            - If dict, keys are node names and values are per-node floats.

        n_jobs: int
            Number of parallel jobs for parameter estimation.
            -1 uses all available cores. Default is 1.

        Returns
        -------
        list of LinearGaussianCPD
            One CPD per node in the model.

        Raises
        ------
        ValueError
            If B0 has wrong size or V0 has wrong shape for any node.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.estimators import LinearGaussianBayesianEstimator
        >>> np.random.seed(42)
        >>> x = np.random.randn(200)
        >>> y = 3.0 * x + np.random.randn(200) * 0.5
        >>> data = pd.DataFrame({"X": x, "Y": y})
        >>> model = LinearGaussianBayesianNetwork([("X", "Y")])
        >>> estimator = LinearGaussianBayesianEstimator(model, data)

        Default flat priors:
        >>> estimator.get_parameters()  # doctest: +ELLIPSIS
        [<LinearGaussianCPD: P(X) = N(-0.041; 0.929) at 0x..., <LinearGaussianCPD: P(Y | X) = N(3.049*X + 0.045; 0.502) at 0x...] # noqa: E501

        Per-node dict priors:
        >>> estimator.get_parameters(
        ...     B0={"X": np.array([0.0]), "Y": np.array([0.0, 0.0])},
        ...     V0={"X": np.eye(1) * 10, "Y": np.eye(2) * 10},
        ... )  # doctest: +ELLIPSIS
        [<LinearGaussianCPD: P(X) = N(-0.041; 0.929) at 0x..., <LinearGaussianCPD: P(Y | X) = N(3.049*X + 0.045; 0.502) at 0x...] # noqa: E501

        """

        def _get_node_param(node):

            parents = sorted(self.model.get_parents(node))
            k = len(parents) + 1

            _B0 = B0[node] if isinstance(B0, dict) else B0
            _V0 = V0[node] if isinstance(V0, dict) else V0
            _alpha_0 = alpha_0[node] if isinstance(alpha_0, dict) else alpha_0
            _beta_0 = beta_0[node] if isinstance(beta_0, dict) else beta_0

            # default priors if not provided
            if _B0 is None:
                _B0 = np.zeros(k)

            if _V0 is None:
                _V0 = np.eye(k) * 10

            if len(np.asarray(_B0).reshape(-1)) != k:
                raise ValueError(
                    f"B0 for node '{node}' has wrong size. "
                    f"Expected {k} (intercept + {len(parents)} parents), got {len(_B0)}."
                )
            if np.asarray(_V0).shape != (k, k):
                raise ValueError(
                    f"V0 for node '{node}' has wrong shape. "
                    f"Expected ({k},{k}), got {np.asarray(_V0).shape}."
                )

            return self.estimate_cpd(
                node,
                B0=_B0,
                V0=_V0,
                alpha_0=_alpha_0,
                beta_0=_beta_0,
            )

        parameters = Parallel(n_jobs=n_jobs)(
            delayed(_get_node_param)(node) for node in self.model.nodes()
        )

        return parameters

    def estimate_cpd(
        self,
        node: str,
        B0: Optional[np.ndarray] = None,
        V0: Optional[np.ndarray] = None,
        alpha_0: float = 2.0,
        beta_0: float = 1.0,
    ) -> LinearGaussianCPD:
        """
        Estimates the LinearGaussianCPD for a single node using
        Normal-Inverse-Gamma conjugate priors.

        Parameters
        ----------
        node: str
            The node whose CPD is to be estimated. Must be present
            in the model.

        B0: np.ndarray or None
            Prior mean vector for beta. Shape must be (k,) where
            k = 1 + number_of_parents. If None, defaults to np.zeros(k).

        V0: np.ndarray or None
            Prior covariance matrix for beta. Shape must be (k, k).
            If None, defaults to np.eye(k) * 10.

        alpha_0: float
            Shape parameter of the Inverse-Gamma prior on variance.
            Must be > 1 for a finite posterior mean. Default is 2.0.

        beta_0: float
            Scale parameter of the Inverse-Gamma prior on variance.
            Default is 1.0.

        Returns
        -------
        LinearGaussianCPD
            The estimated CPD for the given node.

        Raises
        ------
        ValueError
            If B0 has wrong size or V0 has wrong shape for this node.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.estimators import LinearGaussianBayesianEstimator
        >>> np.random.seed(42)
        >>> n = 30
        >>> x = np.random.randn(n)
        >>> y = 2.0 * x + 1.0 + np.random.randn(n) * 0.5
        >>> df = pd.DataFrame({"x1": x, "x2": y})
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2")])
        >>> estimator = LinearGaussianBayesianEstimator(model, df)

        Flat prior (converges to MLE):
        >>> estimator.estimate_cpd("x2")  # doctest: +ELLIPSIS
        <LinearGaussianCPD: P(x2 | x1) = N(2.042*x1 + 0.944; 0.522) at 0x...

        Informative prior pulling intercept toward 10:
        >>> estimator.estimate_cpd(
        ...     "x2",
        ...     B0=np.array([10.0, 0.0]),
        ...     V0=np.eye(2) * 0.5,
        ... )  # doctest: +ELLIPSIS
        <LinearGaussianCPD: P(x2 | x1) = N(2.016*x1 + 1.508; 2.306) at 0x...

        """
        parents = sorted(self.model.get_parents(node))
        k = len(parents) + 1

        if B0 is None:
            B0 = np.zeros(k)

        if V0 is None:
            V0 = np.eye(k) * 10

        if len(np.asarray(B0).reshape(-1)) != k:
            raise ValueError(
                f"B0 for node '{node}' has wrong size. "
                f"Expected {k} (intercept + {len(parents)} parents), got {len(B0)}."
            )
        if np.asarray(V0).shape != (k, k):
            raise ValueError(
                f"V0 for node '{node}' has wrong shape. "
                f"Expected ({k},{k}), got {np.asarray(V0).shape}."
            )

        y = self.data[node].values
        n = len(y)

        if len(parents) == 0:
            X = np.ones((n, 1))
        else:
            X = np.column_stack([np.ones(n)] + [self.data[p].values for p in parents])

        p = X.shape[1]

        # reshape priors
        B0 = np.asarray(B0).reshape(p)
        V0 = np.asarray(V0)

        if V0.shape != (p, p):
            raise ValueError("V0 must be (p,p)")

        # matrices
        XtX = X.T @ X
        Xty = X.T @ y
        V0_inv = np.linalg.inv(V0)

        # Posterior covariance
        Vn = np.linalg.inv(V0_inv + XtX)

        # Posterior mean
        Bn = Vn @ (V0_inv @ B0 + Xty)

        # Variance posterior
        alpha_n = alpha_0 + n / 2

        term1 = y.T @ y
        term2 = B0.T @ V0_inv @ B0
        term3 = Bn.T @ (V0_inv + XtX) @ Bn

        beta_n = beta_0 + 0.5 * (term1 + term2 - term3)

        sigma_sq = beta_n / (alpha_n - 1)

        beta = Bn

        return LinearGaussianCPD(
            variable=node, beta=beta, std=np.sqrt(sigma_sq), evidence=parents
        )
