from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.causal_discovery.bivariate_scores import BaseBivariateScore, get_bivariate_score
from pgmpy.utils import get_dataset_type


class IGCI(BaseCausalDiscovery):
    """
    Bivariate causal discovery using Information-Geometric Causal Inference (IGCI)
    :cite:p:`mooij_2016,janzing_2012`.

    Given two continuous variables, IGCI orients the edge between them under the
    deterministic, invertible model ``Y = f(X)``, assuming:

    - No unobserved confounders.
    - ``Y = f(X)`` is monotonic and close to deterministic (little noise).
    - The cause distribution and ``f`` are independent.

    IGCI scores each direction and orients toward the lower score. If a score
    is ``NaN`` or the directional scores are equal, the method raises a
    :class:`ValueError` because it cannot determine a causal direction.

    Parameters
    ----------
    scoring_method : str, BaseBivariateScore instance, or callable, default="slope"
        Score used to compare the two directions. Use ``"slope"``, ``"entropy"``, a configured
        score object, or a callable of the form ``score(x, y) -> float``.

    ref_measure : {"uniform", "gaussian"}, default="uniform"
        Method used to normalize both variables before scoring. ``"uniform"`` uses min-max
        scaling and ``"gaussian"`` uses standardization.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        Learned graph with the single oriented edge.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix of ``causal_graph_``.

    forward_score_ : float
        Score when the first column is treated as cause. Lower is better.

    backward_score_ : float
        Score when the second column is treated as cause. Lower is better.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import IGCI
    >>> rng = np.random.default_rng(42)
    >>> x = rng.uniform(0, 1, 500)
    >>> df = pd.DataFrame({"X": x, "Y": x**3 + rng.normal(0, 1e-3, 500)})
    >>> igci = IGCI().fit(df)
    >>> list(igci.causal_graph_.edges())
    [('X', 'Y')]
    >>> round(float(igci.forward_score_), 5)
    0.22245
    >>> round(float(igci.backward_score_), 5)
    1.66886
    >>> round(float(IGCI(scoring_method="entropy").fit(df).forward_score_), 5)
    -0.70337

    References
    ----------
    - :cite:p:`mooij_2016`
    - :cite:p:`janzing_2012`

    """

    def __init__(
        self,
        scoring_method: str
        | BaseBivariateScore
        | Callable[[np.typing.ArrayLike, np.typing.ArrayLike], float] = "slope",
        ref_measure: str = "uniform",
    ) -> None:
        self.scoring_method = scoring_method
        self.ref_measure = ref_measure

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    def _fit(self, X: pd.DataFrame) -> "IGCI":
        """
        Orient the edge between the two variables in ``X`` using IGCI.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from. Must contain exactly
            two continuous variables.

        Returns
        -------
        self : IGCI
            Returns the instance with the fitted attributes set.
        """
        # Step 1: Validate hyperparameters and resolve the score function.
        if self.ref_measure not in ("uniform", "gaussian"):
            raise ValueError(f"ref_measure must be one of ('uniform', 'gaussian'). Got: {self.ref_measure!r}")
        score = get_bivariate_score(self.scoring_method, algorithm="igci")

        # Step 2: Validate the input data.
        if X.shape[1] != 2:
            raise ValueError(f"IGCI requires exactly two variables, got {X.shape[1]}.")
        if get_dataset_type(X) != "continuous":
            raise ValueError("IGCI requires continuous (numeric) variables; got non-continuous data.")

        x, y = self.feature_names_in_
        for col in (x, y):
            if X[col].std() == 0:
                raise ValueError(f"Variable '{col}' is constant; IGCI requires non-constant variables.")

        # Step 3: Affine normalization to the reference measure.
        x_vals = X[x].to_numpy(dtype=float)
        y_vals = X[y].to_numpy(dtype=float)
        if self.ref_measure == "uniform":
            x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
            y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())
        else:
            x_norm = (x_vals - x_vals.mean()) / x_vals.std()
            y_norm = (y_vals - y_vals.mean()) / y_vals.std()

        # Step 4: Score both directions.
        forward_score = score(x_norm, y_norm)
        backward_score = score(y_norm, x_norm)

        if np.isnan(forward_score) or np.isnan(backward_score):
            raise ValueError("IGCI scoring_method returned NaN; cannot determine a causal direction.")
        if forward_score == backward_score:
            raise ValueError(
                f"IGCI could not determine a causal direction because both directions produced the same score: "
                f"{forward_score!r}."
            )

        self.forward_score_ = forward_score
        self.backward_score_ = backward_score

        # Step 5: Orient the edge and store the fitted attributes.
        edge = (x, y) if self.forward_score_ < self.backward_score_ else (y, x)
        self.causal_graph_ = DAG([edge])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, nodelist=[x, y], weight=None, dtype="int")

        return self
