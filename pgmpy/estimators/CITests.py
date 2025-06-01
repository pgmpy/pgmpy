import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma as gamma_func
from scipy.spatial.distance import pdist
from sklearn.cross_decomposition import CCA

from pgmpy.global_vars import logger
from pgmpy.independencies import IndependenceAssertion


def _ensure_matrix(data):
    """Ensure data is 2D numpy array."""
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def _normalize(data):
    """Normalize data to have mean 0 and std 1."""
    data = np.asarray(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (data - mean) / std


def _random_fourier_features(x, num_f, sigma, seed=None):
    """
    Generate random Fourier features for kernel approximation.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Input data
    num_f : int
        Number of random features to generate
    sigma : float
        Kernel bandwidth parameter
    seed : int, optional
        Random seed

    Returns
    -------
    dict with 'feat' : array of shape (n_samples, num_f)
        Random Fourier features
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, n_features = x.shape

    # Ensure sigma is not too small to avoid numerical issues
    sigma = max(sigma, 1e-6)

    # Generate random frequencies from normal distribution
    W = np.random.normal(0, 1 / sigma, size=(n_features, num_f))
    # Generate random phase shifts
    b = np.random.uniform(0, 2 * np.pi, size=num_f)

    # Compute random features
    feat = np.sqrt(2.0 / num_f) * np.cos(x @ W + b)

    return {"feat": feat}


def _satterthwaite_welch(eigenvalues, test_stat):
    """Satterthwaite-Welch approximation for weighted chi-squared."""
    c1 = np.sum(eigenvalues)
    c2 = np.sum(eigenvalues**2)

    if c2 <= 0:
        return 0

    # Gamma distribution parameters
    alpha = c1**2 / c2
    beta = c2 / c1

    # P-value using gamma CDF
    p_value = stats.gamma.sf(test_stat, a=alpha, scale=beta)
    return p_value


def _hall_buckley_eagleson(eigenvalues, test_stat):
    """Hall-Buckley-Eagleson approximation."""
    c1 = np.sum(eigenvalues)
    c2 = np.sum(eigenvalues**2)
    c3 = np.sum(eigenvalues**3)

    if c2 <= 0:
        return 0

    h = c2**3 / c3**2
    p_value = stats.chi2.sf(test_stat * h / c2, h)
    return p_value


def _simplified_lpb_approx(eigenvalues, test_stat):
    """
    Simplified Lindsay-Pilla-Basak (LPB) method using gamma approximation.
    This is not the full LPB mixture model, but a simplified version using
    4 moments and gamma approximation.
    """
    try:
        # Calculate first 4 cumulants
        c1 = np.sum(eigenvalues)
        c2 = 2 * np.sum(eigenvalues**2)
        c3 = 8 * np.sum(eigenvalues**3)
        c4 = 48 * np.sum(eigenvalues**4)

        if c2 <= 0:
            return _hall_buckley_eagleson(eigenvalues, test_stat)

        # Simplified LPB using gamma approximation based on skewness and kurtosis
        skew = c3 / (c2**1.5)
        kurt = c4 / (c2**2)

        # Use method of moments for gamma distribution
        alpha = 4 / skew**2
        beta = c2 / c1 * skew / 2

        p_value = stats.gamma.sf(test_stat, a=alpha, scale=beta)
        return p_value

    except:
        # Fallback to HBE if calculation fails
        return _hall_buckley_eagleson(eigenvalues, test_stat)


def _unconditional_rff_test(x_data, y_data, num_f2, approx, seed=None):
    """
    Unconditional independence test using RFF (equivalent to RIT in R code).
    """
    r = x_data.shape[0]
    r1 = min(500, r)

    # Normalize data
    x = _normalize(x_data)
    y = _normalize(y_data)

    # Compute bandwidths
    x_dist = pdist(x[:r1])
    y_dist = pdist(y[:r1])
    sigma_x = np.median(x_dist) if len(x_dist) > 0 else 1
    sigma_y = np.median(y_dist) if len(y_dist) > 0 else 1

    # Ensure bandwidths are not too small
    sigma_x = max(sigma_x, 1e-6)
    sigma_y = max(sigma_y, 1e-6)

    # Generate RFFs with different seeds for independence
    four_x = _random_fourier_features(x, num_f2, sigma_x, seed)
    four_y = _random_fourier_features(
        y, num_f2, sigma_y, seed + 1 if seed is not None else None
    )

    # Normalize features
    f_x = _normalize(four_x["feat"])
    f_y = _normalize(four_y["feat"])

    # Compute test statistic
    f_x_centered = f_x - np.mean(f_x, axis=0)
    f_y_centered = f_y - np.mean(f_y, axis=0)
    Cxy = (1 / r) * (f_x_centered.T @ f_y_centered)
    Sta = r * np.sum(Cxy**2)

    # Compute null distribution
    d1, d2 = np.meshgrid(range(num_f2), range(num_f2))
    d1, d2 = d1.flatten(), d2.flatten()
    res = f_x_centered[:, d1] * f_y_centered[:, d2]
    Cov = (res.T @ res) / r

    # Get eigenvalues with error handling
    try:
        eig_vals = np.linalg.eigvalsh(Cov)
        eig_vals = eig_vals[eig_vals > 1e-10]  # Keep only positive eigenvalues
    except np.linalg.LinAlgError:
        # If eigenvalue computation fails, use a simpler approximation
        # Return high p-value indicating independence
        return Sta, 1.0

    if len(eig_vals) == 0:
        return Sta, 1.0

    # Compute p-value
    if num_f2 == 1:
        approx = "hbe"

    if approx == "gamma":
        p_value = _satterthwaite_welch(eig_vals, Sta)
    elif approx == "hbe":
        p_value = _hall_buckley_eagleson(eig_vals, Sta)
    elif approx == "lpd4":
        p_value = _simplified_lpb_approx(eig_vals, Sta)
    else:
        p_value = _simplified_lpb_approx(eig_vals, Sta)

    return Sta, p_value


def _median_heuristic(data_array):
    """
    Calculate the median heuristic for kernel bandwidth selection.

    Parameters
    ----------
    data_array : np.ndarray
        Data array of shape (n_samples, n_features)

    Returns
    -------
    float
        Median of pairwise Euclidean distances
    """
    n_samples = data_array.shape[0]

    # Handle edge cases
    if n_samples <= 1:
        return 1.0

    if n_samples > 1000:
        # For large datasets, use a subset for efficiency
        indices = np.random.choice(n_samples, size=1000, replace=False)
        data_subset = data_array[indices]
    else:
        data_subset = data_array

    # Check if data has any variance
    if np.all(np.std(data_subset, axis=0) == 0):
        return 1.0

    # Compute pairwise distances
    distances = []
    for i in range(len(data_subset)):
        for j in range(i + 1, len(data_subset)):
            dist = np.linalg.norm(data_subset[i] - data_subset[j])
            if dist > 0:  # Only include non-zero distances
                distances.append(dist)

    # Return median, but ensure it's never 0
    if not distances:
        return 1.0

    median_dist = np.median(distances)
    return max(median_dist, 1e-6)  # Ensure minimum value


def rcit(
    X,
    Y,
    Z,
    data,
    boolean=True,
    significance_level=0.05,
    approx="lpd4",
    num_f=100,
    num_f2=5,
    seed=None,
    **kwargs,
):
    """
    Randomized Conditional Independence Test (RCIT).

    Tests the null hypothesis that X ⊥ Y | Z using kernel methods with
    Random Fourier Features approximation. All variables are transformed
    to RFF space.

    Parameters
    ----------
    X : str or list of str
        Variable name(s) for X
    Y : str or list of str
        Variable name(s) for Y
    Z : str or list of str
        Variable name(s) for conditioning set Z
    data : pd.DataFrame
        The data containing the variables
    boolean : bool
        If True, returns boolean result. If False, returns (statistic, p_value)
    significance_level : float
        Significance level for the test
    approx : str
        Method for approximating null distribution: "lpd4", "gamma", "hbe"
    num_f : int
        Number of features for conditioning set
    num_f2 : int
        Number of features for non-conditioning sets
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    bool or tuple
        If boolean=True: True if independent, False otherwise
        If boolean=False: (test_statistic, p_value)

    References
    ----------
    Strobl, Eric V., et al. "Approximate kernel-based conditional independence
    tests for fast non-parametric causal discovery." Journal of Causal Inference 7.1 (2019).
    """
    # Handle significance_level in kwargs
    if "significance_level" in kwargs:
        significance_level = kwargs["significance_level"]

    # Convert to lists
    X = [X] if isinstance(X, str) else list(X)
    Y = [Y] if isinstance(Y, str) else list(Y)
    Z = [Z] if isinstance(Z, str) else list(Z) if Z else []

    # Extract data
    x_data = _ensure_matrix(data[X])
    y_data = _ensure_matrix(data[Y])

    # Check if x or y have zero variance (constant variables)
    if np.std(x_data) == 0 or np.std(y_data) == 0:
        if boolean:
            return True
        else:
            return 0, 1.0

    # If no conditioning set, use unconditional RFF test
    if len(Z) == 0:
        stat, p_value = _unconditional_rff_test(x_data, y_data, num_f2, approx, seed)
        if boolean:
            return p_value >= significance_level
        else:
            return stat, p_value

    # Extract and prepare data
    z_data = _ensure_matrix(data[Z])

    # Remove constant columns from z
    z_std = np.std(z_data, axis=0)
    z_data = z_data[:, z_std > 0]
    if z_data.shape[1] == 0:
        # No valid conditioning variables - use unconditional test
        stat, p_value = _unconditional_rff_test(x_data, y_data, num_f2, approx, seed)
        if boolean:
            return p_value >= significance_level
        else:
            return stat, p_value

    # Check if x or y have zero variance
    if np.std(x_data) == 0 or np.std(y_data) == 0:
        if boolean:
            return True
        else:
            return 0, 1

    r = x_data.shape[0]  # number of samples
    r1 = min(500, r)  # for distance calculation

    # Normalize data
    x = _normalize(x_data)
    y = _normalize(y_data)
    z = _normalize(z_data)

    # Note: We combine y and z before computing RFF for y.
    # This may differ from the strict interpretation of the Strobl et al. paper where
    # Y's RFFs should be independent of Z's RFFs.
    y_combined = np.hstack([y, z])

    # Compute kernel bandwidths using median heuristic
    z_dist = pdist(z[:r1])
    x_dist = pdist(x[:r1])
    y_dist = pdist(y_combined[:r1])

    sigma_z = np.median(z_dist) if len(z_dist) > 0 else 1
    sigma_x = np.median(x_dist) if len(x_dist) > 0 else 1
    sigma_y = np.median(y_dist) if len(y_dist) > 0 else 1

    # Generate Random Fourier Features with different seeds for independence
    try:
        four_z = _random_fourier_features(z, num_f, sigma_z, seed)
        four_x = _random_fourier_features(
            x, num_f2, sigma_x, seed + 1 if seed is not None else None
        )
        four_y = _random_fourier_features(
            y_combined, num_f2, sigma_y, seed + 2 if seed is not None else None
        )
    except Exception:
        # iff RFF generation fails, return independence
        if boolean:
            return True
        else:
            return 0, 1.0

    # Normalize features
    f_x = _normalize(four_x["feat"])
    f_y = _normalize(four_y["feat"])
    f_z = _normalize(four_z["feat"])

    # Center features for covariance calculations
    f_x_centered = f_x - np.mean(f_x, axis=0)
    f_y_centered = f_y - np.mean(f_y, axis=0)
    f_z_centered = f_z - np.mean(f_z, axis=0)

    # Compute covariance matrices using direct matrix multiplication
    Cxy = (1 / r) * (f_x_centered.T @ f_y_centered)
    Czz = (1 / r) * (f_z_centered.T @ f_z_centered)
    Cxz = (1 / r) * (f_x_centered.T @ f_z_centered)
    Czy = (1 / r) * (f_z_centered.T @ f_y_centered)

    # Compute inverse with regularization
    i_Czz = np.linalg.inv(Czz + np.eye(num_f) * 1e-10)

    # Compute conditional covariance
    Cxy_z = Cxy - Cxz @ i_Czz @ Czy

    # Test statistic
    Sta = r * np.sum(Cxy_z**2)

    # Compute residuals for null distribution
    z_i_Czz = f_z @ i_Czz
    e_x_z = z_i_Czz @ Cxz.T
    e_y_z = z_i_Czz @ Czy

    res_x = f_x - e_x_z
    res_y = f_y - e_y_z

    # Create grid for computing covariance of residuals
    d1, d2 = np.meshgrid(range(num_f2), range(num_f2))
    d1, d2 = d1.flatten(), d2.flatten()
    res = res_x[:, d1] * res_y[:, d2]
    Cov = (res.T @ res) / r

    # Get eigenvalues for null distribution
    eig_vals = np.linalg.eigvalsh(Cov)
    eig_vals = eig_vals[eig_vals > 0]  # Keep only positive eigenvalues

    # Compute p-value based on approximation method
    if num_f2 == 1:
        approx = "hbe"

    if approx == "gamma":
        p_value = _satterthwaite_welch(eig_vals, Sta)
    elif approx == "hbe":
        p_value = _hall_buckley_eagleson(eig_vals, Sta)
    elif approx == "lpd4":
        p_value = _simplified_lpb_approx(eig_vals, Sta)
    else:
        # Default to simplified LPB
        p_value = _simplified_lpb_approx(eig_vals, Sta)

    # Ensure p-value is in valid range
    p_value = np.clip(p_value, 0, 1)

    if boolean:
        return p_value >= significance_level
    else:
        return Sta, p_value


def rcot(
    X,
    Y,
    Z,
    data,
    boolean=True,
    significance_level=0.05,
    approx="lpd4",
    num_f=100,
    seed=None,
    **kwargs,
):
    """
    Randomized conditional Correlation Test (RCoT).

    Tests the null hypothesis that X ⊥ Y | Z by testing for zero partial
    correlation after non-linearly transforming Z using Random Fourier Features.
    Only Z is transformed to RFF space; X and Y remain in original space.

    Parameters
    ----------
    X : str or list of str
        Variable name(s) for X
    Y : str or list of str
        Variable name(s) for Y
    Z : str or list of str
        Variable name(s) for conditioning set Z
    data : pd.DataFrame
        The data containing the variables
    boolean : bool
        If True, returns boolean result. If False, returns (statistic, p_value)
    significance_level : float
        Significance level for the test
    approx : str
        Method for approximating null distribution: "lpd4", "gamma", "hbe"
    num_f : int
        Number of features for conditioning set
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    bool or tuple
        If boolean=True: True if independent, False otherwise
        If boolean=False: (test_statistic, p_value)

    References
    ----------
    Strobl, Eric V., et al. "Approximate kernel-based conditional independence
    tests for fast non-parametric causal discovery." Journal of Causal Inference 7.1 (2019).
    """
    # Handle significance_level in kwargs
    if "significance_level" in kwargs:
        significance_level = kwargs["significance_level"]

    # Convert to lists
    X = [X] if isinstance(X, str) else list(X)
    Y = [Y] if isinstance(Y, str) else list(Y)
    Z = [Z] if isinstance(Z, str) else list(Z) if Z else []

    # Extract data
    x_data = _ensure_matrix(data[X])
    y_data = _ensure_matrix(data[Y])

    # If no conditioning set, use unconditional RFF test
    if len(Z) == 0:
        # For RCoT with no conditioning, we use unconditional test with original data
        # but still use RFF framework for consistency
        stat, p_value = _unconditional_rff_test(x_data, y_data, 5, approx, seed)
        if boolean:
            return p_value >= significance_level
        else:
            return stat, p_value

    # Extract and prepare data
    z_data = _ensure_matrix(data[Z])

    # Remove constant columns from z
    z_std = np.std(z_data, axis=0)
    z_data = z_data[:, z_std > 0]
    if z_data.shape[1] == 0:
        # No valid conditioning variables - use unconditional test
        stat, p_value = _unconditional_rff_test(x_data, y_data, 5, approx, seed)
        if boolean:
            return p_value >= significance_level
        else:
            return stat, p_value

    # Check if x or y have zero variance
    if np.std(x_data) == 0 or np.std(y_data) == 0:
        if boolean:
            return True
        else:
            return 0, 1

    r = x_data.shape[0]  # number of samples
    r1 = min(500, r)  # for distance calculation

    # Normalize data
    x = _normalize(x_data)
    y = _normalize(y_data)
    z = _normalize(z_data)

    # Compute kernel bandwidth for z using median heuristic
    z_dist = pdist(z[:r1]) if z.shape[0] > 1 else np.array([1.0])
    sigma_z = max(np.median(z_dist) if len(z_dist) > 0 else 1, 1e-6)

    # Generate Random Fourier Features only for Z
    try:
        four_z = _random_fourier_features(z, num_f, sigma_z, seed)
        f_z = _normalize(four_z["feat"])
    except Exception:
        if boolean:
            return True
        else:
            return 0, 1.0

    # Use original (normalized) x and y
    px = x.shape[1]
    py = y.shape[1]

    # Center all data for covariance calculations
    x_centered = x - np.mean(x, axis=0)
    y_centered = y - np.mean(y, axis=0)
    f_z_centered = f_z - np.mean(f_z, axis=0)

    # Compute covariance matrices using direct matrix multiplication
    Cxy = (1 / r) * (x_centered.T @ y_centered)
    Czz = (1 / r) * (f_z_centered.T @ f_z_centered)
    Cxz = (1 / r) * (x_centered.T @ f_z_centered)
    Czy = (1 / r) * (f_z_centered.T @ y_centered)

    # Compute inverse with regularization
    try:
        i_Czz = np.linalg.inv(Czz + np.eye(num_f) * 1e-10)
    except np.linalg.LinAlgError:
        if boolean:
            return True
        else:
            return 0, 1.0

    # Compute conditional covariance
    Cxy_z = Cxy - Cxz @ i_Czz @ Czy

    # Test statistic
    Sta = r * np.sum(Cxy_z**2)

    # Compute residuals for null distribution
    z_i_Czz = f_z @ i_Czz
    e_x_z = z_i_Czz @ Cxz.T
    e_y_z = z_i_Czz @ Czy

    res_x = x - e_x_z
    res_y = y - e_y_z

    # Compute covariance of residuals
    d1, d2 = np.meshgrid(range(px), range(py))
    d1, d2 = d1.flatten(), d2.flatten()
    res = res_x[:, d1] * res_y[:, d2]
    Cov = (res.T @ res) / r

    # Get eigenvalues with error handling
    try:
        eig_vals = np.linalg.eigvalsh(Cov)
        eig_vals = eig_vals[eig_vals > 1e-10]
    except np.linalg.LinAlgError:
        if boolean:
            return True
        else:
            return 0, 1.0

    if len(eig_vals) == 0:
        if boolean:
            return True
        else:
            return 0, 1.0

    # Compute p-value
    if px == 1 and py == 1:
        approx = "hbe"

    if approx == "gamma":
        p_value = _satterthwaite_welch(eig_vals, Sta)
    elif approx == "hbe":
        p_value = _hall_buckley_eagleson(eig_vals, Sta)
    elif approx == "lpd4":
        p_value = _simplified_lpb_approx(eig_vals, Sta)
    else:
        p_value = _simplified_lpb_approx(eig_vals, Sta)

    p_value = np.clip(p_value, 0, 1)

    if boolean:
        return p_value >= significance_level
    else:
        return Sta, p_value


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
        "rcit": rcit,
        "rcot": rcot,
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
    Checks if `X ⊥ Y | Z` is in `independencies`. This method is implemented to
    have an uniform API when the independencies are provided instead of data.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X ⊥ Y | Z

    Y: str
        The second variable for testing the independence condition X ⊥ Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X ⊥ Y | Z

    data: pandas.DataFrame The dataset in which to test the independence condition.

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
        higher value), given the null hypothesis that X ⊥ Y | Zs is True.
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
        higher value), given the null hypothesis that X ⊥ Y | Zs is True.
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
        If boolean=True, an additional argument `significance_level` must be         specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False.  If
        boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    CI Test Results: tuple or bool
        If boolean = False, Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X ⊥ Y | Zs is True.
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
        higher value), given the null hypothesis that X ⊥ Y | Zs is True.
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
        higher value), given the null hypothesis that X ⊥ Y | Zs is True.
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
                        f"Skipping the test {X} ⊥ {Y} | {Z[0]}={z_state}. Not enough samples"
                    )
                else:
                    z_str = ", ".join(
                        [f"{var}={state}" for var, state in zip(Z, z_state)]
                    )
                    logger.info(
                        f"Skipping the test {X} ⊥ {Y} | {z_str}. Not enough samples"
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
    Should be used only on continuous data. In case when :math:`Z != \\null` uses
    linear regression and computes pearson coefficient on residuals.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X ⊥ Y | Z

    Y: str
        The second variable for testing the independence condition X ⊥ Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X ⊥ Y | Z

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
        The first variable for testing the independence condition X ⊥ Y | Z

    Y: str
        The second variable for testing the independence condition X ⊥ Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X ⊥ Y | Z

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
        The first variable for testing the independence condition X ⊥ Y | Z

    Y: str
        The second variable for testing the independence condition X ⊥ Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X ⊥ Y | Z

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
