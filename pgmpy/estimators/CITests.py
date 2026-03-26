import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tools.sm_exceptions as sm_exceptions
from scipy import stats
from sklearn.cross_decomposition import CCA

from pgmpy import logger
from pgmpy.independencies import IndependenceAssertion
from pgmpy.utils import get_dataset_type


class CITestRegistry:
    """
    Registry to manage Conditional Independence (CI) Test Strategies.

    Allows looking up tests by name or inferring suitable tests based on data type.
    """

    def __init__(self):
        self._registry: dict[str, Callable] = {}
        self._tags: dict[str, list[str]] = {}
        self._defaults: dict[str, str] = {
            "continuous": "pearsonr",
            "discrete": "chi_square",
            "mixed": "pillai",
        }

    def register(self, name: str, data_types: list[str]):
        """
        Decorator to register a CI test strategy.

        Parameters
        ----------
        name : str
            The name of the test (case-insensitive).

        data_types : list of str
            List of data types this test supports (e.g., ['continuous', 'discrete']).
        """

        def decorator(func: Callable):
            clean_name = name.lower()
            self._registry[clean_name] = func
            self._tags[clean_name] = data_types

            return func

        return decorator

    def list_all(self, data_type=None) -> list[str]:
        """
        Lists all registered CI test strategies.

        Parameters
        ----------
        data_type : str, optional
            If provided, filters tests that support the given data type.

        Returns
        -------
        list of str
            Names of all registered CI tests.
        """
        if data_type:
            return [name for name, types in self._tags.items() if data_type in types]

        return list(self._registry.keys())

    def get_test(self, test: str | None | Callable, data: pd.DataFrame | None = None) -> Callable:
        """
        Retrieves a CI test strategy.

        Parameters
        ----------
        test : str, callable or None
            The name of the test, a callable function, or None.

        data : pandas.DataFrame, optional
            The dataframe used to infer the test type if `test` is None.

        Returns
        -------
        callable
            The CI test function.

        Raises
        ------
        ValueError
            If `test` is None and `data` is None, or if the test name is not found.
        """
        # Case 1: Test is already a function/strategy
        if callable(test):
            return test

        # Case 2: Test is None, infer from data
        if test is None:
            if data is None:
                raise ValueError("Cannot determine a suitable CI test as data is None. Please specify CI test to use.")
            var_type = get_dataset_type(data)
            test_name = self._defaults.get(var_type)
            return self._registry[test_name]

        # Case 3: Test is a string name
        if isinstance(test, str):
            clean_name = test.lower()
            if clean_name in self._registry:
                return self._registry[clean_name]
            else:
                raise ValueError(
                    f"`ci_test` must either be one of {list(self._registry.keys())}, or a callable. Got: {test}"
                )


ci_registry = CITestRegistry()


@ci_registry.register(
    name="independence_match",
    data_types=["discrete", "continuous", "mixed"],
)
def independence_match(X, Y, Z, independencies, **kwargs):
    """
    Check if `X \u27c2 Y | Z` is in `independences`.

    This method is implemented to have a uniform API when the independences
    are provided explicitly instead of being inferred from data.

    Parameters
    ----------
    X : str
        The first variable for testing the independence condition X \u27c2 Y | Z.

    Y : str
        The second variable for testing the independence condition X \u27c2 Y | Z.
    Z : list or array-like
        A list of conditional variables for testing the condition X \u27c2 Y | Z.
    independencies : pgmpy.independencies.Independencies
        The object containing the known independences.

    Returns
    -------
    bool
        True if the independence assertion is present in `independences`, else False.
    """
    warnings.warn(
        "`independence_match` is deprecated. Please use `pgmpy.ci_tests.IndependenceMatch` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return IndependenceAssertion(X, Y, Z) in independencies


@ci_registry.register(name="pearsonr", data_types=["continuous"])
def pearsonr(X, Y, Z, data, boolean=True, **kwargs):
    """
    Compute Pearson correlation coefficient and p-value for testing non-correlation.

    Should be used only on continuous data. In case when :math:`Z \\neq \\emptyset` uses
    linear regression and computes pearson coefficient on residuals.

    Parameters
    ----------
    X : str
        The first variable for testing the independence condition X \u27c2 Y | Z.

    Y : str
        The second variable for testing the independence condition X \u27c2 Y | Z.

    Z : list or array-like
        A list of conditional variables for testing the condition X \u27c2 Y | Z.

    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    boolean : bool, default=True
        If True, returns a boolean indicating independence (based on `significance_level`).
        If False, returns the test statistic and p-value.

    **kwargs
        Additional arguments. Must contain `significance_level` if `boolean=True`.

    Returns
    -------
    result : bool or tuple
        If boolean=True, returns True if p-value >= significance_level, else False.
        If boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    .. [2] https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    """
    warnings.warn(
        "`pearsonr` is deprecated. Please use `pgmpy.ci_tests.Pearsonr` instead.",
        FutureWarning,
        stacklevel=2,
    )
    # Step 1: Test if the inputs are correct
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}")

    # Step 2: If Z is empty compute a non-conditional test.
    if len(Z) == 0:
        coef, p_value = stats.pearsonr(data.loc[:, X], data.loc[:, Y])

    # Step 3: If Z is non-empty, use linear regression to compute residuals and test independence on it.
    else:
        X_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, X], rcond=None)[0]
        Y_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, Y], rcond=None)[0]

        residual_X = data.loc[:, X] - data.loc[:, Z].dot(X_coef)
        residual_Y = data.loc[:, Y] - data.loc[:, Z].dot(Y_coef)
        coef, p_value = stats.pearsonr(residual_X, residual_Y)

    if boolean:
        return p_value >= kwargs["significance_level"]
    else:
        return coef, p_value


@ci_registry.register(name="power_divergence", data_types=["discrete"])
def power_divergence(X, Y, Z, data, boolean=True, lambda_="cressie-read", **kwargs):
    """
    Computes the Cressie-Read power divergence statistic [1]. The null hypothesis
    for the test is X is independent of Y given Z. A lot of the frequency comparision
    based statistics (eg. chi-square, G-test etc) belong to power divergence family,
    and are special cases of this test.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    lambda_: float or string
        The lambda parameter for the power_divergence statistic. Some values of
        lambda_ results in other well known tests:

            * "pearson"             1          "Chi-squared test"
            * "log-likelihood"      0          "G-test or log-likelihood"
            * "freeman-tuckey"     -1/2        "Freeman-Tuckey Statistic"
            * "mod-log-likelihood"  -1         "Modified Log-likelihood"
            * "neyman"              -2         "Neyman's statistic"
            * "cressie-read"        2/3        "The value recommended in the paper[1]"

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    **kwargs
        Must contain `significance_level` if `boolean=True`.

    Returns
    -------
    result : bool or tuple
        If boolean=False, returns (chi, p_value, dof).
        If boolean=True, returns True if p_value > significance_level.

    References
    ----------
    .. [1] Cressie, Noel, and Timothy RC Read. "Multinomial goodness‐of‐fit tests."
      Journal of the Royal Statistical Society: Series B (Methodological) 46.3 (1984): 440-464.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> chi_square(X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05)
    np.True_
    >>> chi_square(
    ...     X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> chi_square(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_

    """
    warnings.warn(
        "`power_divergence` is deprecated. Please use `pgmpy.ci_tests.PowerDivergence` instead.",
        FutureWarning,
        stacklevel=2,
    )
    # Step 1: Check if the arguments are valid and type conversions.
    if hasattr(Z, "__iter__"):
        Z = list(Z)
    else:
        raise (f"Z must be an iterable. Got object type: {type(Z)}")

    if (X in Z) or (Y in Z):
        raise ValueError(f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z.")

    # Step 2: Do a simple contingency test if there are no conditional variables.
    if len(Z) == 0:
        chi, p_value, dof, expected = stats.chi2_contingency(
            data.groupby([X, Y], observed=False).size().unstack(Y, fill_value=0),
            lambda_=lambda_,
        )

    # Step 3: If there are conditionals variables, iterate over unique states
    else:
        chi = 0
        dof = 0
        for z_state, df in data.groupby(Z, observed=True):
            # Compute the contingency table
            unique_x, x_inv = np.unique(df[X], return_inverse=True)
            unique_y, y_inv = np.unique(df[Y], return_inverse=True)
            contingency = np.bincount(x_inv * len(unique_y) + y_inv, minlength=len(unique_x) * len(unique_y)).reshape(
                len(unique_x), len(unique_y)
            )

            # If all values of a column in the contingency table are zeros, skip the test.
            if any(contingency.sum(axis=0) == 0) or any(contingency.sum(axis=1) == 0):
                if isinstance(z_state, str):
                    logger.info(f"Skipping the test {X} _|_ {Y} | {Z[0]}={z_state}. Not enough samples")
                else:
                    z_str = ", ".join([f"{var}={state}" for var, state in zip(Z, z_state)])
                    logger.info(f"Skipping the test {X} _|_ {Y} | {z_str}. Not enough samples")
            else:
                c, _, d, _ = stats.chi2_contingency(contingency, lambda_=lambda_)
                chi += c
                dof += d
        p_value = 1 - stats.chi2.cdf(chi, df=dof)

    # Step 4: Return the values
    if boolean:
        return p_value >= kwargs["significance_level"]
    else:
        return chi, p_value, dof


@ci_registry.register(name="chi_square", data_types=["discrete"])
def chi_square(X, Y, Z, data, boolean=True, **kwargs):
    """
    Perform Chi-square conditional independence test.

    Tests the null hypothesis that X is independent from Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
        be specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False.
        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    result : bool or tuple
        If boolean=False, returns (chi, p_value, dof).
        If boolean=True, returns True if p_value > significance_level.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chi-squared_test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> chi_square(X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05)
    np.True_
    >>> chi_square(
    ...     X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> chi_square(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_
    """
    warnings.warn(
        "`chi_square` is deprecated. Please use `pgmpy.ci_tests.ChiSquare` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return power_divergence(X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="pearson", **kwargs)


@ci_registry.register(name="g_sq", data_types=["discrete"])
def g_sq(X, Y, Z, data, boolean=True, **kwargs):
    """
    G squared test for conditional independence. Also commonly known as G-test,
    likelihood-ratio or maximum likelihood statistical significance test.
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must be
        specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False. If
        boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    result : bool or tuple
        If boolean=False, returns (chi, p_value, dof).
        If boolean=True, returns True if p_value > significance_level.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/G-test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> g_sq(X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05)
    np.True_
    >>> g_sq(X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05)
    np.True_
    >>> g_sq(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_
    """
    warnings.warn(
        "`g_sq` is deprecated. Please use `pgmpy.ci_tests.GSq` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return power_divergence(X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="log-likelihood", **kwargs)


@ci_registry.register(name="log_likelihood", data_types=["discrete"])
def log_likelihood(X, Y, Z, data, boolean=True, **kwargs):
    """
    Log likelihood ratio test for conditional independence. Also commonly known
    as G-test, G-squared test or maximum likelihood statistical significance
    test.  Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must be
        specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False.  If
        boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    CI Test Results: tuple or bool
        If boolean = False, Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X \u27c2 Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/G-test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> log_likelihood(
    ...     X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> log_likelihood(
    ...     X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> log_likelihood(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_
    """
    warnings.warn(
        "`log_likelihood` is deprecated. Please use `pgmpy.ci_tests.LogLikelihood` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return power_divergence(X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="log-likelihood", **kwargs)


@ci_registry.register(name="modified_log_likelihood", data_types=["discrete"])
def modified_log_likelihood(X, Y, Z, data, boolean=True, **kwargs):
    """
    Modified log likelihood ratio test for conditional independence.
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must be
        specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False.
        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    CI Test Results: tuple or bool
        If boolean = False, Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X \u27c2 Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> modified_log_likelihood(
    ...     X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> modified_log_likelihood(
    ...     X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> modified_log_likelihood(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_
    """
    warnings.warn(
        "`modified_log_likelihood` is deprecated. Please use `pgmpy.ci_tests.ModifiedLogLikelihood` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return power_divergence(
        X=X,
        Y=Y,
        Z=Z,
        data=data,
        boolean=boolean,
        lambda_="mod-log-likelihood",
        **kwargs,
    )


def _get_predictions(X, Y, Z, data, **kwargs):
    """
    Helper Strategy: Function to get predictions using XGBoost for `ci_pillai`.
    Not registered directly as a CI test.
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as e:
        raise ImportError(
            f"{e}. xgboost is required for using pillai_trace test. Please install using: pip install xgboost"
        ) from None

    if any(data.loc[:, Z].dtypes == "category"):
        enable_categorical = True
    else:
        enable_categorical = False

    # Helper to fit and predict
    def fit_predict(target_col):
        is_cat = data.loc[:, target_col].dtype == "category"
        model_cls = XGBClassifier if is_cat else XGBRegressor
        model = model_cls(
            enable_categorical=enable_categorical,
            seed=kwargs.get("seed"),
            random_state=kwargs.get("seed"),
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


@ci_registry.register(name="pillai", data_types=["discrete", "continuous", "mixed"])
def pillai_trace(X, Y, Z, data, boolean=True, **kwargs):
    """
    A mixed-data residualization based conditional independence test[1].

    Uses XGBoost estimator to compute LS residuals[2], and then does an
    association test (Pillai's Trace) on the residuals.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \u27c2 Y | Z

    Y: str
        The second variable for testing the independence condition X \u27c2 Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \u27c2 Y | Z

    data: pandas.DataFrame
        The dataset in which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the pearson correlation coefficient and p_value
            of the test.

    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True if p-value >= significance_level, else False. If
        boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value)

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
    warnings.warn(
        "`pillai_trace` is deprecated. Please use `pgmpy.ci_tests.PillaiTrace` instead.",
        FutureWarning,
        stacklevel=2,
    )
    # Step 1: Test if the inputs are correct
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}")

    # Step 1.1: If no conditional variables are specified, use a constant value.
    if len(Z) == 0:
        Z = ["cont_Z"]
        data = data.assign(cont_Z=np.ones(data.shape[0]))

    # Step 2: Get the predictions
    pred_x, pred_y, x_cat_index, y_cat_index = _get_predictions(X, Y, Z, data, **kwargs)

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

    # Step 6: Return
    if boolean:
        return p_value >= kwargs["significance_level"]
    else:
        return coef, p_value


@ci_registry.register(name="gcm", data_types=["continuous"])
def gcm(X, Y, Z, data, boolean=True, **kwargs):
    """
    The Generalized Covariance Measure(GCM) test for CI.

    It performs linear regressions on the conditioning variable and then tests
    for a vanishing covariance between the resulting residuals. Details of the
    method can be found in [1].

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \u27c2 Y | Z

    Y: str
        The second variable for testing the independence condition X \u27c2 Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \u27c2 Y | Z

    data: pandas.DataFrame
        The dataset in which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the pearson correlation coefficient and p_value
            of the test.

    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True if p-value >= significance_level, else False. If
        boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value)

    References
    ----------
    .. [1] Rajen D. Shah, and Jonas Peters. "The Hardness of Conditional Independence Testing and the Generalised
        Covariance Measure".
    """
    warnings.warn(
        "`gcm` is deprecated. Please use `pgmpy.ci_tests.GCM` instead.",
        FutureWarning,
        stacklevel=2,
    )
    # Step 1: Test if the inputs are correct
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}")

    # Step 1.1: Add another column with constant values to handle intercepts.
    Z_aug = Z + ["intercept"]
    data_aug = data.assign(intercept=np.ones(data.shape[0]))

    # Step 2: Compute the linear regression and the residuals
    X_coef = np.linalg.lstsq(data_aug.loc[:, Z_aug], data_aug.loc[:, X], rcond=None)[0]
    Y_coef = np.linalg.lstsq(data_aug.loc[:, Z_aug], data_aug.loc[:, Y], rcond=None)[0]
    res_x = data_aug.loc[:, X] - data_aug.loc[:, Z_aug].dot(X_coef)
    res_y = data_aug.loc[:, Y] - data_aug.loc[:, Z_aug].dot(Y_coef)

    # Step 3: Compute the Generalised Covariance Measure.
    n = res_x.shape[0]
    t_stat = (1 / np.sqrt(n)) * np.dot(res_x, res_y) / np.std(res_x * res_y)

    # Step 4: Compute p-value using standard normal distribution.
    p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

    # Step 6: Return
    if boolean:
        return p_value >= kwargs["significance_level"]
    else:
        return t_stat, p_value


@ci_registry.register(name="pearsonr_equivalence", data_types=["continuous"])
def pearsonr_equivalence(X, Y, Z, data, boolean=True, delta_threshold=0.1, **kwargs) -> tuple | bool:
    """
    Computes a two-sided level-alpha equivalent test using partial correlations.

    Tests the Null Hypothesis that the partial correlation is greater than or
    equal to `delta_threshold` (Dependence). Rejection implies Practical Independence.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X _|_ Y | Z

    Y: str
        The second variable for testing the independence condition X _|_ Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X _|_ Y | Z

    data: pandas.DataFrame
        The dataset in which to test the independence condition.

    boolean: bool
        If True, returns True (Independent) if p_value < significance_level.

    delta_threshold: float
        The equivalence bound (threshold for practical independence).

    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True (Independent) if p-value < significance_level.
        If boolean=False, returns (Partial Correlation, p-value).

    References
    ----------
    .. [1] Malinsky, Daniel. "A cautious approach to constraint-based causal model selection." arXiv preprint
            arXiv:2404.18232 (2024).
    """
    warnings.warn(
        "`pearsonr_equivalence` is deprecated. Please use `pgmpy.ci_tests.PearsonrEquivalence` instead.",
        FutureWarning,
        stacklevel=2,
    )
    # Step 1: Input validation
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}")

    # Step 2: Compute Partial Pearson Correlation and clip values to avoid infinities
    rho, _ = pearsonr(X, Y, Z, data, boolean=False)
    rho = np.clip(rho, -0.999999, 0.999999)

    # Step 3: Fisher Z-Transformation
    coeff = np.arctanh(rho)
    z_delta = np.arctanh(delta_threshold)

    n = data.shape[0]
    s = len(Z)  # Number of conditioning variables

    std_error_factor = np.sqrt(n - s - 3)

    # Step 4: TOST (Two One-Sided Tests)
    # Step 4.1: H0: rho <= -delta  vs  H1: rho > -delta
    z_score_lower = std_error_factor * (coeff + z_delta)
    p_value_lower = 1 - stats.norm.cdf(z_score_lower)

    # Step 4.2: H0: rho >= delta   vs  H1: rho < delta
    z_score_upper = std_error_factor * (coeff - z_delta)
    p_value_upper = stats.norm.cdf(z_score_upper)

    p_value = max(p_value_lower, p_value_upper)

    # Step 5: Return
    if boolean:
        return p_value < kwargs.get("significance_level", 0.05)
    else:
        return coeff, p_value


def _is_categorical(series):
    """
    Determine if a pandas Series should be treated as categorical.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    bool
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if series.dtype.kind in ("O", "S", "U", "b"):
        return True
    return False


def _encode_features(data, columns):
    """
    Build a numeric predictor matrix from a list of column names.
    Continuous columns are kept as-is; categorical columns are
    one-hot encoded with ``drop_first=True``.

    Parameters
    ----------
    data : pd.DataFrame
    columns : list of str

    Returns
    -------
    np.ndarray of shape (n_samples, n_encoded_features)
    """
    if len(columns) == 0:
        return np.empty((len(data), 0))

    parts = []
    for col in columns:
        series = data[col]
        if _is_categorical(series):
            dummies = pd.get_dummies(series, prefix=col, drop_first=True)
            parts.append(dummies.values.astype(float))
        else:
            parts.append(series.values.astype(float).reshape(-1, 1))

    return np.column_stack(parts)


@ci_registry.register(name="regression_based_lr", data_types=["discrete", "continuous", "mixed"])
def regression_based_lr(X, Y, Z, data, boolean=True, **kwargs):
    """
    Regression-based likelihood-ratio conditional independence test.

    Tests the null hypothesis :math:`X \\perp Y \\mid Z` by comparing two nested
    regression models via a likelihood-ratio (or equivalent F-) test.
    The regression family is chosen automatically based on the data type
    of *X*:

    * **Continuous X** -- OLS linear regression, F-test.
    * **Binary X** -- Binary logistic regression, chi-squared LR test.
    * **Categorical X (>2 levels)** -- Multinomial logistic regression, chi-squared LR test.

    Predictor variables (*Y* and *Z*) can be of any type; categorical
    predictors are one-hot encoded internally. This makes the test
    suitable for mixed (continuous + discrete) data.

    Parameters
    ----------
    X : str
        First variable name (used as the response variable in the regression).
    Y : str
        Second variable name.
    Z : list (default: ``[]``)
        A list of variable names for the conditioning set. May be empty.
    data : pd.DataFrame
        The dataset. Columns with dtype ``'object'``, ``'category'``, or
        ``'bool'`` are treated as categorical; all others as continuous.
    boolean : bool (default: ``True``)
        If ``True``, returns ``True`` when X and Y are judged independent
        (p-value >= significance_level), ``False`` otherwise.
        If ``False``, returns a tuple ``(statistic, p_value, dof)``.
    significance_level : float (default: 0.05)
        Significance threshold. Only used when ``boolean=True``.

    Returns
    -------
    bool
        If ``boolean=True``.
    tuple (float, float, int)
        ``(statistic, p_value, dof)`` if ``boolean=False``.

    Notes
    -----
    This test is **asymmetric** -- X is used as the response variable in the
    regression, so the model family is determined by the data type of X.
    Swapping X and Y may yield different test statistics and occasionally
    different p-values, though the independence verdict (True/False) is
    typically consistent. When X and Y have different types (e.g., one
    continuous and one categorical), place the categorical variable as X
    for the most appropriate model selection. If a symmetric test is desired
    for purely continuous data, consider ``pearsonr`` or ``pillai`` instead.

    References
    ----------
    .. [1] Tsagris, M., Borboudakis, G., Lagani, V., & Tsamardinos, I.
       (2018). Constraint-based causal discovery with mixed data.
       *International Journal of Data Science and Analytics*, 6(1),
       19-30. https://doi.org/10.1007/s41060-018-0097-y

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators.CITests import regression_based_lr
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> Z_val = rng.standard_normal(n)
    >>> X_val = 2 * Z_val + rng.standard_normal(n)
    >>> Y_val = 3 * Z_val + rng.standard_normal(n)  # X ind. Y | Z
    >>> data = pd.DataFrame({"X": X_val, "Y": Y_val, "Z": Z_val})
    >>> regression_based_lr("X", "Y", ["Z"], data)
    True
    >>> stat, p, dof = regression_based_lr("X", "Y", [], data, boolean=False)
    >>> p < 0.05          # marginally dependent
    True
    """
    significance_level = kwargs.get("significance_level", 0.05)

    # ---- input validation ------------------------------------------------
    if X not in data.columns:
        raise ValueError(f"Variable '{X}' not found in data columns.")
    if Y not in data.columns:
        raise ValueError(f"Variable '{Y}' not found in data columns.")
    if X == Y:
        raise ValueError(f"X and Y must be different variables, got '{X}' for both.")

    if Z is None:
        Z = []
    if isinstance(Z, str):
        Z = [Z]
    Z = list(Z)

    missing_z = [z for z in Z if z not in data.columns]
    if missing_z:
        raise ValueError(f"Conditioning variable(s) {missing_z} not found in data columns.")

    # Drop rows with any NaN in the relevant columns
    relevant_cols = [X, Y] + Z
    data = data[relevant_cols].dropna()
    n = len(data)

    if n == 0:
        raise ValueError("No valid observations remain after dropping missing values.")

    # ---- encode predictors -----------------------------------------------
    Z_enc = _encode_features(data, Z)  # (n, q_z)
    Y_enc = _encode_features(data, [Y])  # (n, q_y)
    q_y = Y_enc.shape[1]  # extra params contributed by Y

    # Build exogenous matrices (always include intercept)
    if Z_enc.shape[1] > 0:
        restricted_exog = np.column_stack([np.ones((n, 1)), Z_enc])
        full_exog = np.column_stack([np.ones((n, 1)), Z_enc, Y_enc])
    else:
        restricted_exog = np.ones((n, 1))  # intercept only
        full_exog = np.column_stack([np.ones((n, 1)), Y_enc])

    # ---- choose model family based on X type -----------------------------
    x_is_categorical = _is_categorical(data[X])

    if not x_is_categorical:
        # ============ CONTINUOUS X  ->  OLS + F-test =======================
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
            return True if boolean else (0.0, 1.0, df1)

        f_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
        p_value = stats.f.sf(f_stat, df1, df2)

        if boolean:
            return p_value >= significance_level
        return f_stat, p_value, df1

    else:
        # ============ CATEGORICAL X  ->  Logistic / MNLogit + chi^2 ========
        x_encoded, uniques = pd.factorize(data[X])
        n_classes = len(uniques)

        if n_classes < 2:
            # Only one class -- independence is trivially true
            return True if boolean else (0.0, 1.0, 0)

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
            return True if boolean else (0.0, 1.0, 0)

        lr_stat = -2.0 * (model_r.llf - model_f.llf)
        lr_stat = max(lr_stat, 0.0)  # numerical guard

        if dof <= 0:
            return True if boolean else (0.0, 1.0, 0)

        p_value = stats.chi2.sf(lr_stat, dof)

        if boolean:
            return p_value >= significance_level
        return lr_stat, p_value, dof
