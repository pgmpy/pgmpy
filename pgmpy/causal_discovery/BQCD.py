from numbers import Integral

import networkx as nx
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm, rankdata
from sklearn.ensemble import GradientBoostingRegressor

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.utils import get_dataset_type


def _summed_quantile_loss(quantile, observations, tau):
    """Compute the summed quantile loss S_tau(q, z) = (I{q >= z} - tau)(q - z)."""
    quantile = np.asarray(quantile, dtype=float)
    observations = np.asarray(observations, dtype=float)
    return float(np.sum(((quantile >= observations).astype(float) - tau) * (quantile - observations)))


def _rank_to_standard_normal(values):
    """Transform sample ranks to approximate standard-normal values."""
    values = np.asarray(values, dtype=float)
    n = values.size
    ranks = rankdata(values, method="average")
    probs = ranks / (n + 1.0)
    return norm.ppf(probs)


class BQCD(BaseCausalDiscovery):
    """
    Bivariate Quantile Causal Discovery (bQCD) :cite:p:`tagasovska_2020`.

    Given two continuous variables, BQCD orients the edge by comparing
    integrated quantile scores of the marginal and conditional
    distributions in each direction. The preferred causal direction is the one
    with the lower total score. Equal or non-finite scores cannot determine a
    direction.

    Assumptions:

    - The two variables have a causal link, with no confounding, selection bias,
      or feedback.
    - In the causal direction, the marginal distribution of the cause and the
      conditional distribution of the effect given the cause are independent
      mechanisms, making their combined description asymptotically no longer
      than the reverse-direction description.
    - The marginal and conditional quantile estimators are consistent, and the
      code lengths of the fitted quantile models grow sublinearly with the
      sample size.

    This implementation requires exactly two continuous, non-constant input
    variables.

    Parameters
    ----------
    n_quantiles : int, default=3
        Number of Gauss-Legendre nodes used to approximate the integral over
        quantile levels on ``[0, 1]``. The paper recommends three. For fewer
        than 200 samples, one median node is used as in the reference
        implementation.

    quantile_regressor : callable, default=None
        Optional callable ``quantile_regressor(quantile)`` returning a fresh
        regressor for the requested quantile level. If ``None``, each level uses
        ``GradientBoostingRegressor(loss="quantile", alpha=quantile,
        random_state=seed)``.

    seed : int, optional
        Random seed forwarded to the default quantile regressor.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph with the single oriented edge.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of ``causal_graph_``.

    forward_score_ : float
        Normalized ``marginal(X) + conditional(Y | X)`` score after
        Gaussianization and quadrature.

    backward_score_ : float
        Normalized ``marginal(Y) + conditional(X | Y)`` score after
        Gaussianization and quadrature.

    score_components_ : dict
        Integrated marginal and conditional scores used to form both direction
        scores.

    quantile_levels_ : np.ndarray
        Quantile levels used for the integral approximation.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import BQCD
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=200)
    >>> y = x + (0.05 + 2.0 * x**2) * rng.normal(size=200)
    >>> bqcd = BQCD(n_quantiles=3, seed=0).fit(pd.DataFrame({"X": x, "Y": y}))
    >>> list(bqcd.causal_graph_.edges())
    [('X', 'Y')]

    References
    ----------
    - :cite:p:`tagasovska_2020`
    """

    def __init__(self, n_quantiles=3, quantile_regressor=None, seed=None):
        self.n_quantiles = n_quantiles
        self.quantile_regressor = quantile_regressor
        self.seed = seed

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the BQCD algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : BQCD
            Returns the instance with the fitted attributes.
        """
        # Step 1: Validate the configuration and bivariate continuous input.
        if not isinstance(self.n_quantiles, Integral) or isinstance(self.n_quantiles, bool) or self.n_quantiles < 1:
            raise ValueError(f"n_quantiles must be a positive integer, got {self.n_quantiles!r}.")
        if self.quantile_regressor is not None and not callable(self.quantile_regressor):
            raise ValueError("quantile_regressor must be None or callable.")
        if X.shape[1] != 2:
            raise ValueError(f"BQCD requires exactly two variables, got {X.shape[1]}.")

        if get_dataset_type(X) != "continuous":
            raise ValueError("BQCD requires continuous (numeric) variables; got non-continuous data.")

        x_name, y_name = self.feature_names_in_
        for col in (x_name, y_name):
            if X[col].std() == 0:
                raise ValueError(f"Variable '{col}' is constant; BQCD requires non-constant variables.")

        # Step 2: Transform both variables to remove marginal-scale bias.
        x_vals = _rank_to_standard_normal(X[x_name].to_numpy(dtype=float))
        y_vals = _rank_to_standard_normal(X[y_name].to_numpy(dtype=float))
        x_frame = pd.DataFrame({x_name: x_vals})
        y_frame = pd.DataFrame({y_name: y_vals})
        x_series = pd.Series(x_vals, name=x_name)
        y_series = pd.Series(y_vals, name=y_name)

        # Step 3: Map Gauss-Legendre nodes and weights onto [0, 1].
        effective_n_quantiles = 1 if X.shape[0] < 200 else int(self.n_quantiles)
        nodes, weights = leggauss(effective_n_quantiles)
        self.quantile_levels_ = (nodes + 1.0) / 2.0
        self.quadrature_weights_ = weights / 2.0

        # Step 4: Initialize score accumulators and fitted-estimator storage.
        marginal_x = 0.0
        marginal_y = 0.0
        conditional_y_given_x = 0.0
        conditional_x_given_y = 0.0

        self.forward_estimators_ = []
        self.backward_estimators_ = []

        # Step 5: Integrate marginal and conditional scores in both directions.
        for tau, weight in zip(self.quantile_levels_, self.quadrature_weights_, strict=True):
            q_x = np.quantile(x_vals, tau)
            q_y = np.quantile(y_vals, tau)
            marginal_x += weight * _summed_quantile_loss(q_x, x_vals, tau)
            marginal_y += weight * _summed_quantile_loss(q_y, y_vals, tau)

            if self.quantile_regressor is None:
                reg_y_given_x = GradientBoostingRegressor(loss="quantile", alpha=float(tau), random_state=self.seed)
                reg_x_given_y = GradientBoostingRegressor(loss="quantile", alpha=float(tau), random_state=self.seed)
            else:
                reg_y_given_x = self.quantile_regressor(tau)
                reg_x_given_y = self.quantile_regressor(tau)

            reg_y_given_x.fit(x_frame, y_series)
            self.forward_estimators_.append(reg_y_given_x)
            pred_y = np.asarray(reg_y_given_x.predict(x_frame), dtype=float)
            if not np.isfinite(pred_y).all():
                raise ValueError("BQCD quantile regressor produced non-finite predictions.")
            conditional_y_given_x += weight * _summed_quantile_loss(pred_y, y_vals, tau)

            reg_x_given_y.fit(y_frame, x_series)
            self.backward_estimators_.append(reg_x_given_y)
            pred_x = np.asarray(reg_x_given_y.predict(y_frame), dtype=float)
            if not np.isfinite(pred_x).all():
                raise ValueError("BQCD quantile regressor produced non-finite predictions.")
            conditional_x_given_y += weight * _summed_quantile_loss(pred_x, x_vals, tau)

        # Step 6: Store the score components and total directional scores.
        self.score_components_ = {
            "marginal_x": marginal_x,
            "marginal_y": marginal_y,
            "conditional_y_given_x": conditional_y_given_x,
            "conditional_x_given_y": conditional_x_given_y,
        }
        if not np.isfinite(list(self.score_components_.values())).all():
            raise ValueError("BQCD produced non-finite quantile scores.")
        marginal_score = marginal_x + marginal_y
        forward_score = (marginal_x + conditional_y_given_x) / marginal_score
        backward_score = (marginal_y + conditional_x_given_y) / marginal_score
        if forward_score == backward_score:
            raise ValueError(
                "BQCD could not determine a causal direction because both directions produced the same score: "
                f"{forward_score!r}."
            )

        self.forward_score_ = forward_score
        self.backward_score_ = backward_score

        # Step 7: Select the lower-scoring direction and construct its graph.
        edge = (x_name, y_name) if self.forward_score_ < self.backward_score_ else (y_name, x_name)
        self.causal_graph_ = DAG([edge])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, nodelist=[x_name, y_name], weight=None, dtype="int"
        )

        return self
