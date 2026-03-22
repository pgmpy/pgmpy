import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from ._base import _BaseCITest


class GCM(_BaseCITest):
    """
    The Generalized Covariance Measure(GCM) test for CI.

    It fits a regressor on the conditioning variable and then tests for a vanishing covariance between the
    resulting residuals. Details of the method can be found in [1].

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset in which to test the independence condition.
    estimator: optional (default=None)
        Any regressor with fit and predict methods to compute residuals. If None, LinearRegression() is used
        as default.

    Attributes
    ----------
    statistic_ : float
        The GCM t-statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.

    References
    ----------
    .. [1] Rajen D. Shah, and Jonas Peters. "The Hardness of Conditional Independence Testing and the Generalised
        Covariance Measure".
    """

    _tags = {
        "name": "gcm",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, estimator=None):
        self.data = data
        if estimator is None:
            self.estimator = LinearRegression()
        elif not (hasattr(estimator, "fit") and hasattr(estimator, "predict")):
            raise ValueError(f"estimator must have fit and predict methods. Got {type(estimator)} instead.")
        else:
            self.estimator = estimator
        super().__init__()

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute GCM statistic and p-value.

        Sets ``self.statistic_`` (t-statistic) and ``self.p_value_``.
        """
        # Step 1.1: Append intercept column to ensure Z is never empty
        data = self.data
        Z_data = np.column_stack([data.loc[:, list(Z)].values, np.ones(data.shape[0])])

        # Step 2: Compute residuals using the provided estimator
        est_x = clone(self.estimator)
        est_y = clone(self.estimator)
        est_x.fit(Z_data, data.loc[:, X])
        est_y.fit(Z_data, data.loc[:, Y])
        res_x = data.loc[:, X] - est_x.predict(Z_data)
        res_y = data.loc[:, Y] - est_y.predict(Z_data)

        # Step 3: Compute the Generalised Covariance Measure.
        n = res_x.shape[0]
        t_stat = (1 / np.sqrt(n)) * np.dot(res_x, res_y) / np.std(res_x * res_y)

        # Step 4: Compute p-value using standard normal distribution.
        p_value = 2 * stats.norm.sf(np.abs(t_stat))

        self.statistic_ = t_stat
        self.p_value_ = p_value

        return self.statistic_, self.p_value_
