import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA

from pgmpy.global_vars import logger
from pgmpy.independencies import IndependenceAssertion


def get_ci_test(test, full=False, data=None, independencies=None):
    if callable(test):
        return test

    test = test.lower()
    supported_tests = {
        "chi_square": chi_square,
        "g_sq": g_sq,
        "log_likelihood": log_likelihood,
        "modified_log_likelihood": modified_log_likelihood,
        "pearsonr": pearsonr,
        "pillai": pillai_trace,
        "gcm": gcm,
    }
    if full:
        supported_tests["power_divergence"] = power_divergence
        supported_tests["independence_match"] = independence_match

    if test not in supported_tests.keys():
        raise ValueError(
            f"ci_test must either be one of {list(supported_tests.keys())}, or a function. Got: {test}"
        )

    if full:
        if test == "independence_match":
            if independencies is None:
                raise ValueError(
                    "For using independence_match, independencies argument must be specified"
                )
        elif data is None:
            raise ValueError(
                "For using Chi Square or Pearsonr, data argument must be specified"
            )

    return supported_tests[test]


def independence_match(X, Y, Z, independencies, **kwargs):
    """
    Checks if `X \u27c2 Y | Z` is in `independencies`. This method is implemented to
    have an uniform API when the independencies are provided instead of data.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \u27c2 Y | Z

    Y: str
        The second variable for testing the independence condition X \u27c2 Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \u27c2 Y | Z

    data: pandas.DataFrame The dataset in which to test the indepenedence condition.

    Returns
    -------
    p-value: float (Fixed to 0 since it is always confident)
    """
    return IndependenceAssertion(X, Y, Z) in independencies


def chi_square(X, Y, Z, data, boolean=True, **kwargs):
    """
    Chi-square conditional independence test.
    Tests the null hypothesis that X is independent from Y given Zs.

    This is done by comparing the observed frequencies with the expected
    frequencies if X,Y were conditionally independent, using a chisquare
    deviance statistic. The expected frequencies given independence are
    :math:`P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
    as :math:`P(X,Zs)*P(Y,Zs)/P(Zs).

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
    CI Test Results: tuple or bool
        If boolean = False, Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X \u27c2 Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Chi-squared_test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> chi_square(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="pearson", **kwargs
    )


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
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> g_sq(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> g_sq(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> g_sq(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="log-likelihood", **kwargs
    )


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
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> log_likelihood(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> log_likelihood(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> log_likelihood(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="log-likelihood", **kwargs
    )


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
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> modified_log_likelihood(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> modified_log_likelihood(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> modified_log_likelihood(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X,
        Y=Y,
        Z=Z,
        data=data,
        boolean=boolean,
        lambda_="mod-log-likelihood",
        **kwargs,
    )


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
            "pearson"             1          "Chi-squared test"
            "log-likelihood"      0          "G-test or log-likelihood"
            "freeman-tuckey"     -1/2        "Freeman-Tuckey Statistic"
            "mod-log-likelihood"  -1         "Modified Log-likelihood"
            "neyman"              -2         "Neyman's statistic"
            "cressie-read"        2/3        "The value recommended in the paper[1]"

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
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

    References
    ----------
    [1] Cressie, Noel, and Timothy RC Read. "Multinomial goodness‐of‐fit tests." Journal of the Royal Statistical Society: Series B (Methodological) 46.3 (1984): 440-464.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> chi_square(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    # Step 1: Check if the arguments are valid and type conversions.
    if hasattr(Z, "__iter__"):
        Z = list(Z)
    else:
        raise (f"Z must be an iterable. Got object type: {type(Z)}")

    if (X in Z) or (Y in Z):
        raise ValueError(
            f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z."
        )

    # Step 2: Do a simple contingency test if there are no conditional variables.
    if len(Z) == 0:
        chi, p_value, dof, expected = stats.chi2_contingency(
            data.groupby([X, Y], observed=False).size().unstack(Y, fill_value=0),
            lambda_=lambda_,
        )

    # Step 3: If there are conditionals variables, iterate over unique states and do
    #         the contingency test.
    else:
        chi = 0
        dof = 0
        for z_state, df in data.groupby(Z, observed=True):
            # Compute the contingency table
            unique_x, x_inv = np.unique(df[X], return_inverse=True)
            unique_y, y_inv = np.unique(df[Y], return_inverse=True)
            contingency = np.bincount(
                x_inv * len(unique_y) + y_inv, minlength=len(unique_x) * len(unique_y)
            ).reshape(len(unique_x), len(unique_y))

            # If all values of a column in the contingency table are zeros, skip the test.
            if any(contingency.sum(axis=0) == 0) or any(contingency.sum(axis=1) == 0):
                if isinstance(z_state, str):
                    logger.info(
                        f"Skipping the test {X} \u27c2 {Y} | {Z[0]}={z_state}. Not enough samples"
                    )
                else:
                    z_str = ", ".join(
                        [f"{var}={state}" for var, state in zip(Z, z_state)]
                    )
                    logger.info(
                        f"Skipping the test {X} \u27c2 {Y} | {z_str}. Not enough samples"
                    )
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


def pearsonr(X, Y, Z, data, boolean=True, **kwargs):
    """
    Computes Pearson correlation coefficient and p-value for testing non-correlation.
    Should be used only on continuous data. In case when :math:`Z != \null` uses
    linear regression and computes pearson coefficient on residuals.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \u27c2 Y | Z

    Y: str
        The second variable for testing the independence condition X \u27c2 Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \u27c2 Y | Z

    data: pandas.DataFrame
        The dataset in which to test the indepenedence condition.

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
    [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    [2] https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    """
    # Step 1: Test if the inputs are correct
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}"
        )

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
        if p_value >= kwargs["significance_level"]:
            return True
        else:
            return False
    else:
        return coef, p_value


def _get_predictions(X, Y, Z, data, **kwargs):
    """
    Function to get predictions using XGBoost for `ci_pillai`.
    """
    # Step 0: Check if XGboost is installed.
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as e:
        raise ImportError(
            e.msg
            + ". xgboost is required for using pillai_trace test. Please install using: pip install xgboost"
        ) from None

    # Step 1: Check if any of the conditional variables are categorical
    if any(data.loc[:, Z].dtypes == "category"):
        enable_categorical = True
    else:
        enable_categorical = False

    # Step 2: Check variable type of X, choose estimator, and compute predictions.
    if data.loc[:, X].dtype == "category":
        clf_x = XGBClassifier(
            enable_categorical=enable_categorical,
            seed=kwargs.get("seed"),
            random_state=kwargs.get("seed"),
        )
        x, x_cat_index = pd.factorize(data.loc[:, X])
        clf_x.fit(data.loc[:, Z], x)
        pred_x = clf_x.predict_proba(data.loc[:, Z])
    else:
        clf_x = XGBRegressor(
            enable_categorical=enable_categorical,
            seed=kwargs.get("seed"),
            random_state=kwargs.get("seed"),
        )
        x = data.loc[:, X]
        x_cat_index = None
        clf_x.fit(data.loc[:, Z], x)
        pred_x = clf_x.predict(data.loc[:, Z])

    # Step 3: Check variable type of Y, choose estimator, and compute predictions.
    if data.loc[:, Y].dtype == "category":
        clf_y = XGBClassifier(
            enable_categorical=enable_categorical,
            seed=kwargs.get("seed"),
            random_state=kwargs.get("seed"),
        )
        y, y_cat_index = pd.factorize(data.loc[:, Y])
        clf_y.fit(data.loc[:, Z], y)
        pred_y = clf_y.predict_proba(data.loc[:, Z])
    else:
        clf_y = XGBRegressor(
            enable_categorical=enable_categorical,
            seed=kwargs.get("seed"),
            random_state=kwargs.get("seed"),
        )
        y = data.loc[:, Y]
        y_cat_index = None
        clf_y.fit(data.loc[:, Z], y)
        pred_y = clf_y.predict(data.loc[:, Z])

    # Step 4: Return the predictions.
    return (pred_x, pred_y, x_cat_index, y_cat_index)


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
        The dataset in which to test the indepenedence condition.

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
    [1] Ankan, Ankur, and Johannes Textor. "A simple unified approach to testing high-dimensional conditional independences for categorical and ordinal data." Proceedings of the AAAI Conference on Artificial Intelligence.
    [2] Li, C.; and Shepherd, B. E. 2010. Test of Association Between Two Ordinal Variables While Adjusting for Covariates. Journal of the American Statistical Association.
    [3] Muller, K. E. and Peterson B. L. (1984) Practical Methods for computing power in testing the multivariate general linear hypothesis. Computational Statistics & Data Analysis.
    """
    # Step 1: Test if the inputs are correct
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}"
        )

    # Step 1.1: If no conditional variables are specified, use a constant value.
    if len(Z) == 0:
        Z = ["cont_Z"]
        data = data.assign(cont_Z=np.ones(data.shape[0]))

    # Step 2: Get the predictions
    pred_x, pred_y, x_cat_index, y_cat_index = _get_predictions(X, Y, Z, data, **kwargs)

    # Step 3: Compute the residuals
    if data.loc[:, X].dtype == "category":
        x = pd.get_dummies(data.loc[:, X]).loc[
            :, x_cat_index.categories[x_cat_index.codes]
        ]
        # Drop last column to avoid multicollinearity
        res_x = (x - pred_x).iloc[:, :-1]
    else:
        res_x = data.loc[:, X] - pred_x

    if data.loc[:, Y].dtype == "category":
        y = pd.get_dummies(data.loc[:, Y]).loc[
            :, y_cat_index.categories[y_cat_index.codes]
        ]
        # Drop last column to avoid multicollinearity
        res_y = (y - pred_y).iloc[:, :-1]
    else:
        res_y = data.loc[:, Y] - pred_y

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

    # Step 5: Compute p-value using f-approximation [3].
    s = min(res_x.shape[1], res_y.shape[1])
    df1 = res_x.shape[1] * res_y.shape[1]
    df2 = s * (data.shape[0] - 1 + s - res_x.shape[1] - res_y.shape[1])
    f_stat = (coef / df1) * (df2 / (s - coef))
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    # Step 6: Return
    if boolean:
        if p_value >= kwargs["significance_level"]:
            return True
        else:
            return False
    else:
        return coef, p_value


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
        The dataset in which to test the indepenedence condition.

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
    [1] Rajen D. Shah, and Jonas Peters. "The Hardness of Conditional Independence Testing and the Generalised Covariance Measure".
    """
    # Step 1: Test if the inputs are correct
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}"
        )

    # Step 1.1: Add another column with constant values to handle intercepts. When Z=[],
    #           this can act as the constant vector.
    Z += ["intercept"]
    data = data.assign(intercept=np.ones(data.shape[0]))

    # Step 2: Compute the linear regression and the residuals
    X_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, X], rcond=None)[0]
    Y_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, Y], rcond=None)[0]
    res_x = data.loc[:, X] - data.loc[:, Z].dot(X_coef)
    res_y = data.loc[:, Y] - data.loc[:, Z].dot(Y_coef)

    # Step 3: Compute the Generalised Covariance Measure.
    n = res_x.shape[0]
    t_stat = (1 / np.sqrt(n)) * np.dot(res_x, res_y) / np.std(res_x * res_y)

    # Step 4: Compute p-value using standard normal distribution.
    p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

    # Step 6: Return
    if boolean:
        if p_value >= kwargs["significance_level"]:
            return True
        else:
            return False
    else:
        return t_stat, p_value
