import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA

from ._base import _BaseCITest


class PillaiTrace(_BaseCITest):
    """
    A mixed-data residualization based conditional independence test[1].

    Uses XGBoost estimator to compute LS residuals[2], and then does an
    association test (Pillai's Trace) on the residuals.

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset in which to test the independence condition.

    seed : int, optional
        Random seed used for the underlying XGBoost models.

    Attributes
    ----------
    statistic_ : float
        Pillai's trace statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test, computed via F-approximation. Set after calling the test.

    References
    ----------
    .. [1] Ankan, Ankur, and Johannes Textor. "A simple unified approach to testing high-dimensional" "conditional
           independences for categorical and ordinal data." Proceedings of the
           AAAI Conference on Artificial Intelligence.
    .. [2] Li, C.; and Shepherd, B. E. 2010. Test of Association Between Two Ordinal Variables While Adjusting for
           Covariates. Journal of the American Statistical Association.
    .. [3] Muller, K. E. and Peterson B. L. (1984) Practical Methods for computing power in testing the multivariate
           general linear hypothesis. Computational Statistics & Data Analysis.
    """

    _tags = {
        "name": "pillai",
        "data_types": ("discrete", "continuous", "mixed"),
        "default_for": "mixed",
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, seed=None):
        self.seed = seed
        self.data = data
        super().__init__()

    def _get_predictions(self, X, Y, Z, data):
        """
        Get XGBoost predictions for X and Y given Z.
        Uses ``self.seed`` for reproducibility.
        """
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as e:
            raise ImportError(
                f"{e}. xgboost is required for using pillai_trace test. Please install using: pip install xgboost"
            ) from None

        enable_categorical = any(data.loc[:, Z].dtypes == "category")

        def fit_predict(target_col):
            is_cat = data.loc[:, target_col].dtype == "category"
            model_cls = XGBClassifier if is_cat else XGBRegressor
            model = model_cls(
                enable_categorical=enable_categorical,
                seed=self.seed,
                random_state=self.seed,
            )

            target_data = data.loc[:, target_col]
            cat_index = None

            if is_cat:
                y_encoded, cat_index = pd.factorize(target_data)
                model.fit(data.loc[:, Z], y_encoded)
                pred = model.predict_proba(data.loc[:, Z])
            else:
                model.fit(data.loc[:, Z], target_data)
                pred = model.predict(data.loc[:, Z])

            return pred, cat_index

        pred_x, x_cat_index = fit_predict(X)
        pred_y, y_cat_index = fit_predict(Y)

        return pred_x, pred_y, x_cat_index, y_cat_index

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute Pillai's trace statistic and p-value.

        Sets ``self.statistic_`` (Pillai's trace) and ``self.p_value_``.
        """
        data = self.data
        # Step 1.1: If no conditional variables are specified, use a constant value.
        if len(Z) == 0:
            Z = ["cont_Z"]
            data = data.assign(cont_Z=np.ones(data.shape[0]))

        # Step 2: Get the predictions
        pred_x, pred_y, x_cat_index, y_cat_index = self._get_predictions(X, Y, Z, data)

        # Step 3: Compute the residuals
        def get_residuals(col_name, pred, cat_index):
            if data.loc[:, col_name].dtype == "category":
                dummies = pd.get_dummies(data.loc[:, col_name]).loc[:, cat_index.categories[cat_index.codes]]
                # Drop last column to avoid multicollinearity
                return (dummies - pred).iloc[:, :-1]
            else:
                return data.loc[:, col_name] - pred

        res_x = get_residuals(X, pred_x, x_cat_index)
        res_y = get_residuals(Y, pred_y, y_cat_index)

        # Step 4: Compute Pillai's trace.
        if isinstance(res_x, pd.Series):
            res_x = res_x.to_frame()
        if isinstance(res_y, pd.Series):
            res_y = res_y.to_frame()

        cca = CCA(scale=False, n_components=min(res_x.shape[1], res_y.shape[1]))
        res_x_c, res_y_c = cca.fit_transform(res_x, res_y)

        cancor = []
        for i in range(min(res_x.shape[1], res_y.shape[1])):
            cancor.append(np.corrcoef(res_x_c[:, [i]].T, res_y_c[:, [i]].T)[0, 1])

        coef = (np.array(cancor) ** 2).sum()

        # Step 5: Compute p-value using f-approximation.
        s = min(res_x.shape[1], res_y.shape[1])
        df1 = res_x.shape[1] * res_y.shape[1]
        df2 = s * (data.shape[0] - 1 + s - res_x.shape[1] - res_y.shape[1])
        f_stat = (coef / df1) * (df2 / (s - coef))
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        self.statistic_ = coef
        self.p_value_ = p_value

        return self.statistic_, self.p_value_
