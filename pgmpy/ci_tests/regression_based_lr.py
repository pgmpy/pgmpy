import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tools.sm_exceptions as sm_exceptions
from scipy import stats

from pgmpy import logger
from pgmpy.utils import preprocess_data

from ._base import _BaseCITest


def _encode_features(data, columns, dtypes):
    """
    Build a numeric predictor matrix from a list of column names.

    Continuous columns (dtype ``"N"``) are kept as-is; categorical columns
    (dtype ``"C"`` or ``"O"``) are one-hot encoded with ``drop_first=True``
    to avoid multicollinearity with the intercept term.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    columns : list of str
        Column names to encode.
    dtypes : dict
        Mapping of column name to dtype string (``"N"``, ``"C"``, or ``"O"``)
        as returned by :func:`pgmpy.utils.preprocess_data`.

    Returns
    -------
    np.ndarray of shape (n_samples, n_encoded_features)
    """
    if len(columns) == 0:
        return np.empty((len(data), 0))

    parts = []
    for col in columns:
        if dtypes[col] in ("C", "O"):
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            parts.append(dummies.values.astype(float))
        else:
            parts.append(data[col].values.astype(float).reshape(-1, 1))

    return np.column_stack(parts)


class RegressionBasedLR(_BaseCITest):
    r"""
    Regression-based likelihood-ratio conditional independence test.

    Tests the null hypothesis :math:`X \perp Y \mid Z` by comparing two nested
    regression models via a likelihood-ratio (or equivalent F-) test.
    The regression family is chosen automatically based on the data type of *X*
    as inferred by :func:`pgmpy.utils.preprocess_data`:

    * **Continuous X** -- OLS linear regression, F-test.
    * **Binary X** -- Binary logistic regression, chi-squared LR test.
    * **Categorical X (>2 levels)** -- Multinomial logistic regression,
      chi-squared LR test.

    Predictor variables (*Y* and *Z*) of any type are one-hot encoded
    internally via ``pd.get_dummies(drop_first=True)``. This makes the test
    suitable for mixed (continuous + discrete) data.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition. Columns with
        dtype ``object``, ``category``, or ``bool`` are treated as categorical;
        all others are treated as continuous.

    Attributes
    ----------
    statistic_ : float
        The test statistic (F-statistic for continuous X; chi-squared
        likelihood-ratio statistic for categorical X). Set after calling the
        test.
    p_value_ : float
        The p-value for the test. Set after calling the test.
    dof_ : int
        Degrees of freedom for the test. Set after calling the test.

    Notes
    -----
    This test is **asymmetric** -- X is used as the response variable in the
    regression, so the model family is determined by X's data type. Swapping X
    and Y may yield different test statistics and occasionally different
    p-values, though the independence verdict is typically consistent. When X
    and Y have different types, place the categorical variable as X for the
    most appropriate model selection. For a symmetric test on purely continuous
    data, consider :class:`~pgmpy.ci_tests.Pearsonr` instead.

    References
    ----------
    .. [1] Tsagris, M., Borboudakis, G., Lagani, V., & Tsamardinos, I.
       (2018). Constraint-based causal discovery with mixed data.
       *International Journal of Data Science and Analytics*, 6(1), 19-30.
       https://doi.org/10.1007/s41060-018-0097-y

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import RegressionBasedLR
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> Z = rng.standard_normal(n)
    >>> X = 2 * Z + rng.standard_normal(n)
    >>> Y = 3 * Z + rng.standard_normal(n)
    >>> data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    >>> test = RegressionBasedLR(data=data)
    >>> test("X", "Y", ["Z"], significance_level=0.05)
    True
    >>> round(test.p_value_, 2) >= 0.05
    True
    """

    _tags = {
        "name": "regression_based_lr",
        "data_types": ("discrete", "continuous", "mixed"),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame):
        self.data = data
        super().__init__()

    def run_test(self, X: str, Y: str, Z: list):
        """
        Compute the regression-based LR test statistic and p-value.

        Sets ``self.statistic_``, ``self.p_value_``, and ``self.dof_``.

        Parameters
        ----------
        X : str
            First variable (response in the regression).
        Y : str
            Second variable.
        Z : list of str
            Conditioning set. May be empty.
        """
        relevant_cols = [X, Y] + Z
        data = self.data[relevant_cols].dropna()
        n = len(data)

        if n == 0:
            raise ValueError("No valid observations remain after dropping missing values.")

        # Use pgmpy's preprocess_data to classify column types
        _, dtypes = preprocess_data(data)

        # Encode predictors (Y and Z): continuous kept as-is, categorical one-hot
        Z_enc = _encode_features(data, Z, dtypes)  # (n, q_z)
        Y_enc = _encode_features(data, [Y], dtypes)  # (n, q_y)
        q_y = Y_enc.shape[1]  # extra parameters contributed by Y

        # Build design matrices (always include intercept)
        ones = np.ones((n, 1))
        if Z_enc.shape[1] > 0:
            restricted_exog = np.column_stack([ones, Z_enc])
            full_exog = np.column_stack([ones, Z_enc, Y_enc])
        else:
            restricted_exog = ones
            full_exog = np.column_stack([ones, Y_enc])

        x_is_categorical = dtypes[X] in ("C", "O")

        if not x_is_categorical:
            # ---- CONTINUOUS X: OLS + F-test --------------------------------
            x_values = data[X].values.astype(float)

            rank = np.linalg.matrix_rank(full_exog)
            if rank < full_exog.shape[1]:
                logger.warning(
                    f"regression_based_lr: design matrix is rank-deficient "
                    f"({rank} < {full_exog.shape[1]}). Results may be unreliable."
                )

            model_r = sm.OLS(x_values, restricted_exog).fit()
            model_f = sm.OLS(x_values, full_exog).fit()

            rss_r = model_r.ssr
            rss_f = model_f.ssr
            df1 = q_y
            df2 = n - full_exog.shape[1]

            if df1 <= 0 or df2 <= 0 or rss_f <= 0:
                self.statistic_ = 0.0
                self.p_value_ = 1.0
                self.dof_ = df1
                return self.statistic_, self.p_value_

            f_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
            p_value = stats.f.sf(f_stat, df1, df2)

            self.statistic_ = f_stat
            self.p_value_ = p_value
            self.dof_ = df1

        else:
            # ---- CATEGORICAL X: Logistic / MNLogit + chi^2 LR test --------
            x_encoded, uniques = pd.factorize(data[X])
            n_classes = len(uniques)

            if n_classes < 2:
                # Constant response: independence is trivially true
                self.statistic_ = 0.0
                self.p_value_ = 1.0
                self.dof_ = 0
                return self.statistic_, self.p_value_

            try:
                if n_classes == 2:
                    model_r = sm.Logit(x_encoded, restricted_exog).fit(disp=0, method="lbfgs", maxiter=200)
                    model_f = sm.Logit(x_encoded, full_exog).fit(disp=0, method="lbfgs", maxiter=200)
                    dof = q_y
                else:
                    model_r = sm.MNLogit(x_encoded, restricted_exog).fit(disp=0, method="lbfgs", maxiter=200)
                    model_f = sm.MNLogit(x_encoded, full_exog).fit(disp=0, method="lbfgs", maxiter=200)
                    dof = q_y * (n_classes - 1)
            except (np.linalg.LinAlgError, sm_exceptions.PerfectSeparationError) as e:
                logger.warning(
                    f"regression_based_lr: model fitting failed ({type(e).__name__}: {e}). "
                    "Returning independence (conservative)."
                )
                self.statistic_ = 0.0
                self.p_value_ = 1.0
                self.dof_ = 0
                return self.statistic_, self.p_value_

            lr_stat = max(-2.0 * (model_r.llf - model_f.llf), 0.0)

            if dof <= 0:
                self.statistic_ = 0.0
                self.p_value_ = 1.0
                self.dof_ = 0
                return self.statistic_, self.p_value_

            p_value = stats.chi2.sf(lr_stat, dof)

            self.statistic_ = lr_stat
            self.p_value_ = p_value
            self.dof_ = dof

        return self.statistic_, self.p_value_
