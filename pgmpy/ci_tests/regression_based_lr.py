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

    The ``regression_family`` parameter controls which model class is used.
    When set to ``"auto"`` (default), the family is selected from the data type
    of *X* as inferred by :func:`pgmpy.utils.preprocess_data`:

    * **Continuous X** -- OLS linear regression, F-test (``"linear"``).
    * **Binary X** -- Binary logistic regression, chi-squared LR test (``"logistic"``).
    * **Categorical X (>2 levels)** -- Multinomial logistic regression,
      chi-squared LR test (``"multinomial"``).

    Predictor variables (*Y* and *Z*) of any type are one-hot encoded
    internally via ``pd.get_dummies(drop_first=True)``. This makes the test
    suitable for mixed (continuous + discrete) data.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition. Columns with
        dtype ``object``, ``category``, or ``bool`` are treated as categorical;
        all others are treated as continuous.
    regression_family : str, default ``"auto"``
        Regression model to use. One of ``"auto"``, ``"linear"``,
        ``"logistic"``, ``"multinomial"``, or ``"ordinal"``. When ``"auto"``,
        the family is inferred from the data type of *X*. ``"ordinal"`` is
        reserved for a future release and raises ``NotImplementedError``.

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
    >>> from pgmpy.ci_tests import RegressionBasedLR
    >>> from pgmpy.models import LinearGaussianBayesianNetwork
    >>> from pgmpy.factors.continuous import LinearGaussianCPD
    >>> model = LinearGaussianBayesianNetwork([('Z', 'X'), ('Z', 'Y')])
    >>> model.add_cpds(
    ...     LinearGaussianCPD('Z', [0], 1),
    ...     LinearGaussianCPD('X', [0, 2], 1, ['Z']),
    ...     LinearGaussianCPD('Y', [0, 3], 1, ['Z']))
    >>> data = model.simulate(n_samples=500, seed=42)
    >>> test = RegressionBasedLR(data)
    >>> test('X', 'Y', ['Z'], significance_level=0.05)
    True
    """

    _tags = {
        "name": "regression_based_lr",
        "data_types": ("discrete", "continuous", "mixed"),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, regression_family: str = "auto"):
        self.data = data
        if regression_family not in ("auto", "linear", "logistic", "multinomial", "ordinal"):
            raise ValueError(
                f"regression_family must be one of 'auto', 'linear', 'logistic', 'multinomial', or 'ordinal'. "
                f"Got {regression_family!r}."
            )
        if regression_family == "ordinal":
            raise NotImplementedError("Ordinal regression is not yet supported. It will be added in a future release.")
        self.regression_family = regression_family
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
        # Step 1: Drop missing values and classify column data types.
        data = self.data[[X, Y] + Z].dropna()
        n = len(data)

        if n == 0:
            raise ValueError("No valid observations remain after dropping missing values.")

        _, dtypes = preprocess_data(data)

        # Step 2: Encode Y and Z into numeric design matrices (one-hot for categorical).
        Z_enc = _encode_features(data, Z, dtypes)
        Y_enc = _encode_features(data, [Y], dtypes)
        q_y = Y_enc.shape[1]

        ones = np.ones((n, 1))
        if Z_enc.shape[1] > 0:
            restricted_exog = np.column_stack([ones, Z_enc])
            full_exog = np.column_stack([ones, Z_enc, Y_enc])
        else:
            restricted_exog = ones
            full_exog = np.column_stack([ones, Y_enc])

        # Step 3: Determine the regression family from the data type of X if not specified.
        family = self.regression_family
        if family == "auto":
            x_is_categorical = dtypes[X] in ("C", "O")
            if x_is_categorical:
                x_encoded, uniques = pd.factorize(data[X])
                family = "logistic" if len(uniques) == 2 else "multinomial"
            else:
                family = "linear"

        # Step 4: Fit restricted and full models and compute the test statistic.
        if family == "linear":
            x_values = data[X].values.astype(float)

            rank = np.linalg.matrix_rank(full_exog)
            if rank < full_exog.shape[1]:
                logger.warning(
                    f"regression_based_lr: design matrix is rank-deficient "
                    f"({rank} < {full_exog.shape[1]}). Results may be unreliable."
                )

            model_r = sm.OLS(x_values, restricted_exog).fit()
            model_f = sm.OLS(x_values, full_exog).fit()

            df1 = q_y
            df2 = n - full_exog.shape[1]

            if df1 <= 0 or df2 <= 0 or model_f.ssr <= 0:
                self.statistic_, self.p_value_, self.dof_ = 0.0, 1.0, df1
                return self.statistic_, self.p_value_

            # F-test is the exact finite-sample equivalent of the LR test under Gaussian errors.
            f_stat = ((model_r.ssr - model_f.ssr) / df1) / (model_f.ssr / df2)
            self.statistic_ = f_stat
            self.dof_ = df1

        else:
            if family == "logistic":
                x_encoded, _ = pd.factorize(data[X])
                dof = q_y
                try:
                    model_r = sm.Logit(x_encoded, restricted_exog).fit(disp=0, method="lbfgs", maxiter=200)
                    model_f = sm.Logit(x_encoded, full_exog).fit(disp=0, method="lbfgs", maxiter=200)
                except (np.linalg.LinAlgError, sm_exceptions.PerfectSeparationError) as e:
                    logger.warning(
                        f"regression_based_lr: model fitting failed ({type(e).__name__}: {e}). "
                        "Returning independence (conservative)."
                    )
                    self.statistic_, self.p_value_, self.dof_ = 0.0, 1.0, 0
                    return self.statistic_, self.p_value_
            else:
                x_encoded, uniques = pd.factorize(data[X])
                n_classes = len(uniques)

                if n_classes < 2:
                    self.statistic_, self.p_value_, self.dof_ = 0.0, 1.0, 0
                    return self.statistic_, self.p_value_

                dof = q_y * (n_classes - 1)
                try:
                    model_r = sm.MNLogit(x_encoded, restricted_exog).fit(disp=0, method="lbfgs", maxiter=200)
                    model_f = sm.MNLogit(x_encoded, full_exog).fit(disp=0, method="lbfgs", maxiter=200)
                except (np.linalg.LinAlgError, sm_exceptions.PerfectSeparationError) as e:
                    logger.warning(
                        f"regression_based_lr: model fitting failed ({type(e).__name__}: {e}). "
                        "Returning independence (conservative)."
                    )
                    self.statistic_, self.p_value_, self.dof_ = 0.0, 1.0, 0
                    return self.statistic_, self.p_value_

            if dof <= 0:  # pragma: no cover
                self.statistic_, self.p_value_, self.dof_ = 0.0, 1.0, 0
                return self.statistic_, self.p_value_

            self.statistic_ = max(-2.0 * (model_r.llf - model_f.llf), 0.0)
            self.dof_ = dof

        # Step 5: Compute and return the p-value.
        if family == "linear":
            self.p_value_ = float(stats.f.sf(self.statistic_, self.dof_, df2))
        else:
            self.p_value_ = float(stats.chi2.sf(self.statistic_, self.dof_))

        return self.statistic_, self.p_value_
