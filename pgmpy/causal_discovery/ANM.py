from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.causal_discovery.bivariate_scores import BaseBivariateScore, get_bivariate_score
from pgmpy.utils import get_dataset_type


class ANM(BaseCausalDiscovery):
    """
    Bivariate causal discovery using Additive Noise Models (ANM) :cite:p:`hoyer_2008`.

    Given two continuous variables, ANM orients the edge between them under the additive noise model
    ``effect = f(cause) + noise``, assuming:

    - Causal sufficiency (no unobserved confounders);
    - The data follows the additive noise model;
    - The causal direction is identifiable only if the function ``f`` is nonlinear, or if the noise is non-Gaussian.

    The method fits a regressor in both directions and orients the edge toward the one whose residuals are more
    independent of the input. If a score is ``NaN`` or the directional scores are equal, the method raises a
    :class:`ValueError` because it cannot determine a causal direction.

    Parameters
    ----------
    regressor : sklearn regressor, default=None
        Regressor used to estimate ``f``. If ``None``, a :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
        with an RBF plus white-noise kernel is used, with the input and target standardized so that the unit-scale
        kernel initialization is appropriate regardless of the scale of the data.

    scoring_method : str, BaseBivariateScore instance, or callable, default="independence"
        How a candidate direction is scored from the cause and the regression residuals. A smaller score means the
        residuals are more independent of the cause, i.e. a better-fitting direction. One of:

        - a built-in name -- ``"independence"`` (a CI-test effect size, the default;
          :class:`~pgmpy.causal_discovery.bivariate_scores.IndependenceScore` with ``"pearsonr"``), ``"entropy"``
          (:class:`~pgmpy.causal_discovery.bivariate_scores.EntropyScore`), or ``"gauss"``
          (:class:`~pgmpy.causal_discovery.bivariate_scores.GaussScore`);
        - a configured :class:`~pgmpy.causal_discovery.bivariate_scores.BaseBivariateScore` instance, e.g.
          ``EntropyScore(method="vasicek")``;
        - any callable of the form ``fn(cause, residual) -> float``.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph with the single oriented edge.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of ``causal_graph_``.

    forward_score_ : float
        Direction score for the first-column -> second-column direction, as computed by ``scoring_method``. A
        smaller value means the residuals are more independent of the cause.

    backward_score_ : float
        Direction score for the second-column -> first-column direction. The edge is oriented toward whichever
        direction has the smaller score.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import ANM
    >>> from pgmpy.causal_discovery.bivariate_scores import EntropyScore
    >>> rng = np.random.default_rng(42)
    >>> x = rng.uniform(-2, 2, 500)
    >>> df = pd.DataFrame({"X": x, "Y": x**3 + rng.laplace(size=500)})
    >>> anm = ANM().fit(df)
    >>> anm.causal_graph_.edges()
    OutEdgeView([('X', 'Y')])
    >>> round(float(anm.forward_score_), 5)
    0.00104
    >>> round(float(anm.backward_score_), 5)
    0.00543

    The scoring method can be selected by name:

    >>> ANM(scoring_method="entropy").fit(df).causal_graph_.edges()
    OutEdgeView([('X', 'Y')])

    or passed as an object for full control over its hyperparameters:

    >>> ANM(scoring_method=EntropyScore(method="vasicek")).fit(df).causal_graph_.edges()
    OutEdgeView([('X', 'Y')])

    References
    ----------
    - :cite:p:`hoyer_2008`
    - :cite:p:`mooij_2016`

    """

    def __init__(
        self,
        regressor: BaseEstimator | None = None,
        scoring_method: str
        | BaseBivariateScore
        | Callable[[np.typing.ArrayLike, np.typing.ArrayLike], float] = "independence",
    ) -> None:
        self.regressor = regressor
        self.scoring_method = scoring_method

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    def _fit(self, X: pd.DataFrame) -> "ANM":
        """
        The fitting procedure for the ANM algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.ANM
            Returns the instance with the fitted attributes.
        """
        # Step 0: Validate the inputs and initialize attributes.
        if X.shape[1] != 2:
            raise ValueError(f"ANM requires exactly two variables, got {X.shape[1]}.")

        if get_dataset_type(X) != "continuous":
            raise ValueError("ANM requires continuous (numeric) variables; got non-continuous data.")

        x, y = self.feature_names_in_
        for col in (x, y):
            if X[col].std() == 0:
                raise ValueError(f"Variable '{col}' is constant; ANM requires non-constant variables.")

        # Step 1: Fit models in both directions and compute the residual-dependence scores.
        score_fn = get_bivariate_score(self.scoring_method, algorithm="anm")
        forward_score = self._direction_score(cause=X[[x]], effect=X[y], score_fn=score_fn)
        backward_score = self._direction_score(cause=X[[y]], effect=X[x], score_fn=score_fn)

        if np.isnan(forward_score) or np.isnan(backward_score):
            raise ValueError("ANM scoring_method returned NaN; cannot determine a causal direction.")
        if forward_score == backward_score:
            raise ValueError(
                f"ANM could not determine a causal direction because both directions produced the same score: "
                f"{forward_score!r}."
            )

        self.forward_score_ = forward_score
        self.backward_score_ = backward_score

        # Step 2: Orient the edge toward the direction with the smaller score.
        edge = (x, y) if self.forward_score_ < self.backward_score_ else (y, x)
        self.causal_graph_ = DAG([edge])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, nodelist=[x, y], weight=None, dtype="int")

        return self

    def _direction_score(
        self,
        cause: pd.DataFrame,
        effect: pd.Series,
        score_fn: Callable[[np.typing.ArrayLike, np.typing.ArrayLike], float],
    ) -> float:
        """
        Score the residual dependence for the ``cause -> effect`` direction.

        Fits the regressor to predict ``effect`` from ``cause``, then scores how dependent the resulting residuals
        are on ``cause`` using the configured scoring method.

        Parameters
        ----------
        cause : pd.DataFrame
            The candidate cause variable, as a single-column frame (the regressor's 2D input).

        effect : pd.Series
            The candidate effect variable, regressed on ``cause``.

        score_fn : callable
            Resolved score callable with signature ``score_fn(cause, residual) -> float``.

        Returns
        -------
        score : float
            The configured scoring method evaluated on ``cause`` and the residuals. A smaller value indicates more
            independent residuals, i.e. a better-fitting direction.
        """
        if self.regressor is None:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            regressor = make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e1)),
                    normalize_y=True,
                ),
            )
        else:
            regressor = clone(self.regressor)

        regressor.fit(cause, effect)
        residual = effect - regressor.predict(cause)

        return score_fn(np.asarray(cause).ravel(), np.asarray(residual))
