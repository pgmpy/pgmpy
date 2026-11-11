import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.utils import get_dataset_type


class DirectLiNGAM(BaseCausalDiscovery):
    r"""Direct Linear Non-Gaussian Acyclic Model (DirectLiNGAM) algorithm.

    DirectLiNGAM discovers the causal DAG and ordering of continuous variables under a
    linear structural equation model with non-Gaussian continuous error distributions
    :cite:p:`Shimizu2011`:

    .. math::

        X = B X + e

    where :math:`B` is a strictly lower-triangular connection matrix when variables are
    ordered according to the causal topological order, and :math:`e` are mutually
    independent non-Gaussian continuous noise terms.

    The algorithm directly and iteratively discovers exogenous root variables without
    requiring iterative optimization or local search:

    1. In each step, an exogenous root variable :math:`X_m` is identified among the remaining
       candidate variables by evaluating pairwise mutual independence between candidate variables
       and residuals of regressing all other candidate variables on them.
    2. The linear effect of :math:`X_m` is subtracted from all remaining variables via linear
       regression residuals.
    3. Once the complete causal order :math:`K = (k_1, \dots, k_p)` is discovered, each variable
       is regressed on all its predecessors in the causal order to estimate connection matrix
       :math:`B`. Coefficients below ``threshold`` are pruned.

    Parameters
    ----------
    random_state : int, RandomState instance, Generator, or None, default=None
        Seed or random number generator for subsampling in kernel independence calculation.

    measure : {"pwling", "kernel", "mutual_info"}, default="pwling"
        Measure used to evaluate pairwise independence:

        - ``"pwling"``: Pairwise likelihood ratios / mutual information difference based on
          Hyv\xe4rinen and Smith (2013) differential entropy approximation.
        - ``"kernel"``: Hilbert-Schmidt Independence Criterion (HSIC) with Gaussian RBF kernel.
        - ``"mutual_info"``: Continuous mutual information regression.

    threshold : float, default=0.1
        Minimum absolute regression coefficient to include an edge in the learned DAG.
        Coefficients with absolute value below ``threshold`` are pruned to 0.

    Attributes
    ----------
    causal_order_ : list of Hashable
        The estimated causal ordering of variables from root causes to downstream effects.

    causal_graph_ : pgmpy.base.DAG
        The learned directed acyclic graph.

    adjacency_matrix_ : pandas.DataFrame
        Binary adjacency matrix in the input feature order. Directed edges have entry 1 in the
        cause-to-effect entry, and 0 otherwise.

    B_ : pandas.DataFrame
        Estimated connection matrix containing pruned linear regression coefficients, where row
        indicates target (effect) and column indicates predictor (cause). When ordered by
        ``causal_order_``, it is strictly lower triangular.

    n_features_in_ : int
        Number of features in the data used to learn the graph.

    feature_names_in_ : numpy.ndarray
        Names of the features in the data used to learn the graph.

    Time Complexity
    ---------------
    :math:`O(p^2 \cdot n)` when ``measure="pwling"``, where :math:`p` is the number of variables
    and :math:`n` is the number of samples.

    Space Complexity
    ----------------
    :math:`O(p \cdot n)` for observational data and residual computations.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import DirectLiNGAM
    >>> rng = np.random.default_rng(42)
    >>> n = 1000
    >>> x = rng.uniform(-2, 2, n)
    >>> y = 1.5 * x + rng.laplace(0, 1, n)
    >>> z = 0.8 * y + rng.uniform(-1, 1, n)
    >>> data = pd.DataFrame({"X": x, "Y": y, "Z": z})
    >>> model = DirectLiNGAM(random_state=42).fit(data)
    >>> model.causal_order_
    ['X', 'Y', 'Z']
    >>> sorted(model.causal_graph_.edges())
    [('X', 'Y'), ('Y', 'Z')]

    References
    ----------
    - :cite:p:`Shimizu2011` Shimizu, S., Inazumi, T., Sogawa, Y., Hyvärinen, A., Kawahara, Y.,
      Washio, T., Hoyer, P. O., & Bormann, K. (2011). DirectLiNGAM: A direct method for learning
      a linear non-Gaussian structural equation model. Journal of Machine Learning Research, 12,
      1225-1248.
    - :cite:p:`Hyvarinen2013` Hyvärinen, A., & Smith, S. M. (2013). Pairwise Likelihood Ratios
      for Estimation of Non-Gaussian Structural Equation Models. Journal of Machine Learning
      Research, 14, 111-140.
    """

    _tags = {
        "object_type": "causal_discovery",
        "name": "DirectLiNGAM",
        "data_types": ["continuous"],
        "identifiable_graph": "dag",
        "requires_target": False,
        "capability:multivariate": True,
        "capability:expert_knowledge": False,
        "assumption:causal_sufficiency": True,
        "assumption:acyclicity": True,
        "assumption:faithfulness": True,
        "assumption:linearity": True,
        "assumption:additive_noise": True,
        "assumption:gaussian_noise": False,
        "assumption:non_gaussian_noise": True,
        "assumption:low_noise": False,
    }

    def __init__(
        self,
        random_state: int | np.random.RandomState | np.random.Generator | None = None,
        measure: str = "pwling",
        threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.random_state = random_state
        self.measure = measure
        self.threshold = threshold

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    @staticmethod
    def _entropy(u: np.ndarray) -> float:
        """Approximate differential entropy of standardized 1D array u (Hyvärinen & Smith, 2013)."""
        Hv = (1.0 + np.log(2.0 * np.pi)) / 2.0
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457

        u_clipped = np.clip(u, -30.0, 30.0)
        term1 = np.mean(np.log(np.cosh(u_clipped))) - gamma
        term2 = np.mean(u * np.exp(-0.5 * (u**2)))

        return float(Hv - k1 * (term1**2) - k2 * (term2**2))

    def _diff_mutual_info(self, xi_std: np.ndarray, xj_std: np.ndarray, ri_j: np.ndarray, rj_i: np.ndarray) -> float:
        """Calculate the difference of mutual informations for candidate direction."""
        std_ri = np.std(ri_j)
        std_rj = np.std(rj_i)
        ri_j_std = (ri_j - np.mean(ri_j)) / (std_ri if std_ri > 1e-12 else 1.0)
        rj_i_std = (rj_i - np.mean(rj_i)) / (std_rj if std_rj > 1e-12 else 1.0)
        return (self._entropy(xj_std) + self._entropy(ri_j_std)) - (self._entropy(xi_std) + self._entropy(rj_i_std))

    @staticmethod
    def _compute_hsic(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_samples: int = 500) -> float:
        """Compute HSIC (Hilbert-Schmidt Independence Criterion) between x and y."""
        n = len(x)
        if n > max_samples:
            idx = rng.choice(n, size=max_samples, replace=False)
            x = x[idx]
            y = y[idx]
            n = max_samples

        dx = (x[:, None] - x[None, :]) ** 2
        dy = (y[:, None] - y[None, :]) ** 2

        med_x = float(np.median(dx[dx > 0])) if np.any(dx > 0) else 1.0
        med_y = float(np.median(dy[dy > 0])) if np.any(dy > 0) else 1.0

        K = np.exp(-dx / (2.0 * med_x + 1e-12))
        L = np.exp(-dy / (2.0 * med_y + 1e-12))

        Kc = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
        Lc = L - L.mean(axis=0, keepdims=True) - L.mean(axis=1, keepdims=True) + L.mean()

        hsic = np.sum(Kc * Lc) / ((n - 1) ** 2)
        return max(0.0, float(hsic))

    @staticmethod
    def _residual(xi: np.ndarray, xj: np.ndarray, cov_ij: float, var_j: float) -> np.ndarray:
        """The residual when xi is regressed on xj."""
        if var_j <= 1e-12:
            return xi.copy()
        return xi - (cov_ij / var_j) * xj

    def _search_exogenous(self, X_mat: np.ndarray, U: list[int], rng: np.random.Generator) -> int:
        """Identify the exogenous root variable from remaining candidates U."""
        if len(U) == 1:
            return U[0]

        if self.measure == "pwling":
            X_std = {}
            vars_u = {}
            for i in U:
                xi = X_mat[:, i]
                s = float(np.std(xi))
                vars_u[i] = s**2
                X_std[i] = (xi - np.mean(xi)) / (s if s > 1e-12 else 1.0)

            penalties = []
            for i in U:
                M = 0.0
                xi = X_mat[:, i]
                for j in U:
                    if i != j:
                        xj = X_mat[:, j]
                        cov_ij = float(np.cov(xi, xj, bias=True)[0, 1])
                        ri_j = self._residual(xi, xj, cov_ij, vars_u[j])
                        rj_i = self._residual(xj, xi, cov_ij, vars_u[i])
                        diff = self._diff_mutual_info(X_std[i], X_std[j], ri_j, rj_i)
                        M += min(0.0, diff) ** 2
                penalties.append(M)
            return U[int(np.argmin(penalties))]

        elif self.measure == "kernel":
            scores = []
            for i in U:
                M = 0.0
                xi = X_mat[:, i]
                var_i = float(np.var(xi))
                for j in U:
                    if i != j:
                        xj = X_mat[:, j]
                        cov_ij = float(np.cov(xi, xj, bias=True)[0, 1])
                        rj_i = self._residual(xj, xi, cov_ij, var_i)
                        M += self._compute_hsic(xi, rj_i, rng)
                scores.append(M)
            return U[int(np.argmin(scores))]

        elif self.measure == "mutual_info":
            from sklearn.feature_selection import mutual_info_regression

            scores = []
            for i in U:
                M = 0.0
                xi = X_mat[:, i]
                var_i = float(np.var(xi))
                for j in U:
                    if i != j:
                        xj = X_mat[:, j]
                        cov_ij = float(np.cov(xi, xj, bias=True)[0, 1])
                        rj_i = self._residual(xj, xi, cov_ij, var_i)
                        mi = mutual_info_regression(
                            xi[:, None],
                            rj_i,
                            random_state=self.random_state if isinstance(self.random_state, int) else None,
                        )[0]
                        M += float(mi)
                scores.append(M)
            return U[int(np.argmin(scores))]
        else:
            raise ValueError(
                f"Unknown measure: {self.measure!r}. Supported measures are 'pwling', 'kernel', 'mutual_info'."
            )

    def _fit(self, X: pd.DataFrame) -> "DirectLiNGAM":
        """Fit DirectLiNGAM algorithm to find causal order and DAG."""
        if get_dataset_type(X) != "continuous":
            raise ValueError("DirectLiNGAM requires continuous (numeric) variables; got non-continuous data.")

        if self.measure not in ("pwling", "kernel", "mutual_info"):
            raise ValueError(f"measure must be one of 'pwling', 'kernel', 'mutual_info'. Got: {self.measure!r}")

        if self.threshold is not None and self.threshold < 0:
            raise ValueError(f"threshold must be non-negative. Got: {self.threshold}")

        rng = np.random.default_rng(self.random_state)
        n_features = len(self.feature_names_in_)

        if n_features == 1:
            self.causal_order_ = list(self.feature_names_in_)
            self.causal_graph_ = DAG()
            self.causal_graph_.add_nodes_from(self.causal_order_)
            self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
                encoding="binary", nodelist=list(self.feature_names_in_)
            )
            self.B_ = pd.DataFrame(0.0, index=self.feature_names_in_, columns=self.feature_names_in_)
            return self

        X_mat = X.to_numpy(dtype=float, copy=True)
        U = list(range(n_features))
        order_indices = []

        for _ in range(n_features - 1):
            m = self._search_exogenous(X_mat, U, rng)
            order_indices.append(m)

            # Remove effect of X_m on remaining variables by linear regression residuals
            xm = X_mat[:, m]
            var_xm = float(np.var(xm))
            if var_xm > 1e-12:
                for j in U:
                    if j != m:
                        xj = X_mat[:, j]
                        cov_jm = float(np.cov(xj, xm, bias=True)[0, 1])
                        X_mat[:, j] = xj - (cov_jm / var_xm) * xm
            U.remove(m)

        order_indices.append(U[0])
        self.causal_order_ = [self.feature_names_in_[i] for i in order_indices]

        # Compute strictly lower triangular connection matrix B and DAG
        B = pd.DataFrame(0.0, index=self.feature_names_in_, columns=self.feature_names_in_)
        causal_graph = DAG()
        causal_graph.add_nodes_from(self.feature_names_in_)

        for i in range(1, len(self.causal_order_)):
            target = self.causal_order_[i]
            predictors = self.causal_order_[:i]
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X[predictors], X[target])
            for pred, coef in zip(predictors, reg.coef_):
                if abs(coef) >= self.threshold:
                    B.loc[target, pred] = float(coef)
                    causal_graph.add_edge(pred, target, weight=float(coef))

        self.B_ = B
        self.causal_graph_ = causal_graph
        self.adjacency_matrix_ = causal_graph.to_adjacency(encoding="binary", nodelist=list(self.feature_names_in_))
        return self

    def fit(self, X: pd.DataFrame, y=None) -> "DirectLiNGAM":
        """Fit data (`X`) to a causal DAG using DirectLiNGAM.

        Parameters
        ----------
        X : pandas.DataFrame
            The continuous observational training data.
        y : None
            Ignored. Present for API consistency with scikit-learn.

        Returns
        -------
        self : DirectLiNGAM
            Fitted estimator instance with ``causal_order_``, ``causal_graph_``,
            ``adjacency_matrix_``, and ``B_`` set.
        """
        return super().fit(X, y=y)
