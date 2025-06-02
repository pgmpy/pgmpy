"""
Utility functions for Conditional Independence Tests.

This module contains helper functions for kernel-based CI tests including:
- Random Fourier Features generation
- P-value approximation methods (HBE, LPB)
- Data preprocessing utilities
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.special import comb
from scipy.optimize import brentq


# Data preprocessing utilities
def ensure_matrix(data):
    """Ensure data is 2D numpy array."""
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def normalize(data):
    """Normalize data to have mean 0 and std 1."""
    data = np.asarray(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    return (data - mean) / std


# Random Fourier Features
def random_fourier_features(x, num_f, sigma, seed=None):
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

    sigma = max(sigma, 1e-6)

    # Generate random frequencies from normal distribution
    W = np.random.normal(0, 1 / sigma, size=(n_features, num_f))
    # Generate random phase shifts
    b = np.random.uniform(0, 2 * np.pi, size=num_f)

    # Compute random features
    feat = np.sqrt(2.0 / num_f) * np.cos(x @ W + b)

    return {"feat": feat}


def median_heuristic(data_array):
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
            if dist > 0:
                distances.append(dist)

    # Return median
    if not distances:
        return 1.0

    median_dist = np.median(distances)
    return max(median_dist, 1e-6)


# LPB helper functions
def get_cumulant_vec(eigenvalues, p):
    """Compute cumulants kappa_1, ..., kappa_2p for weighted chi-squared."""
    index = np.arange(1, 2 * p + 1)
    kappa = np.zeros(len(index))
    for i in range(len(index)):
        kappa[i] = (
            (2 ** (index[i] - 1))
            * np.math.factorial(index[i] - 1)
            * np.sum(eigenvalues ** index[i])
        )
    return kappa


def get_moments_from_cumulants(cumul_vec):
    """Convert cumulants to moments using the moment-cumulant relationship."""
    moment_vec = np.copy(cumul_vec)
    if len(moment_vec) > 1:
        for n in range(1, len(moment_vec)):
            m = np.arange(1, n + 1)
            sum_terms = np.sum(comb(n, m - 1) * cumul_vec[m - 1] * moment_vec[n - m])
            moment_vec[n] = cumul_vec[n] + sum_terms
    return moment_vec


def get_weighted_sum_of_chi_squared_moments(eigenvalues, p):
    """Get first 2p moments of weighted sum of chi-squared RVs."""
    cumul_vec = get_cumulant_vec(eigenvalues, p)
    moment_vec = get_moments_from_cumulants(cumul_vec)
    return moment_vec


def get_lambdatilde_1(m1, m2):
    """Get initial lambdatilde estimate."""
    return m2 / (m1**2) - 1


def deltaNmat_applied(x, m_vec, N):
    """Compute the delta_N matrix for LPB method."""
    Nplus1 = N + 1
    m_vec_extended = np.append([1], m_vec[0 : (2 * N)])

    coeff_vec = np.append([0], np.arange(0, 2 * N)) * x + 1
    prod_x_terms_vec = 1 / np.cumprod(coeff_vec)

    delta_mat = np.zeros((Nplus1, Nplus1))
    for i in range(Nplus1):
        for j in range(Nplus1):
            index = i + j
            delta_mat[i, j] = m_vec_extended[index] * prod_x_terms_vec[index]
    return delta_mat


def det_deltaNmat(x, m_vec, N):
    """Return determinant of delta_N matrix."""
    return np.linalg.det(deltaNmat_applied(x, m_vec, N))


def get_lambdatilde_p(lambdatilde_1, p, moment_vec, bisect_tol):
    """Compute lambdatilde_p using bisection method."""
    lambdatilde_vec = np.zeros(p)
    lambdatilde_vec[0] = lambdatilde_1

    if p > 1:
        for i in range(1, p):
            try:
                root = brentq(
                    f=det_deltaNmat,
                    a=1e-10,
                    b=lambdatilde_vec[i - 1],
                    args=(moment_vec, i + 1),
                    xtol=bisect_tol,
                )
                lambdatilde_vec[i] = root
            except ValueError:
                # If bisection fails, use previous value
                lambdatilde_vec[i] = lambdatilde_vec[i - 1] * 0.9

    return lambdatilde_vec[p - 1]


def get_Stilde_poly_coeff(M_p):
    """Get polynomial coefficients from matrix M_p."""
    n = M_p.shape[0]
    mu_poly_coeff_vec = np.zeros(n)

    for i in range(n):
        mat_copy = M_p.copy()
        base_vec = np.zeros(n)
        base_vec[i] = 1
        mat_copy[:, n - 1] = base_vec
        mu_poly_coeff_vec[i] = np.linalg.det(mat_copy)

    return mu_poly_coeff_vec


def get_real_poly_roots(mu_poly_coeff_vec):
    """Get real parts of polynomial roots, sorted in increasing order."""
    mu_roots = np.real(np.roots(mu_poly_coeff_vec[::-1]))
    return np.sort(mu_roots)


def get_vandermonde(vec):
    """Generate Vandermonde matrix from vector."""
    p = len(vec)
    vdm = np.zeros((p, p))
    for i in range(p):
        vdm[i] = vec**i
    return vdm


def gen_and_solve_VDM_system(M_p, mu_roots):
    """Generate Vandermonde matrix and solve linear system for mixing weights."""
    b = M_p[:, 0]
    b = b[:-1]

    # Generate Vandermonde matrix
    vdm = get_vandermonde(mu_roots)

    # Solve linear system
    try:
        pi_vec = np.linalg.solve(vdm, b)
    except np.linalg.LinAlgError:
        # If system is singular, use least squares
        pi_vec, _, _, _ = np.linalg.lstsq(vdm, b, rcond=None)

    return pi_vec


def get_mixed_cdf_value(q, mu_vec, pi_vec, lambdatilde_p):
    """Compute mixture of gamma CDFs at point q."""
    p = len(mu_vec)

    # Shape parameter
    alpha = 1 / lambdatilde_p

    # Scale parameters (beta = mu/alpha as per Lindsay formulation)
    beta_vec = mu_vec / alpha

    # Compute weighted sum of gamma CDFs
    cdf_value = 0.0
    for i in range(p):
        if beta_vec[i] > 0:  # Only use positive scale parameters
            cdf_value += pi_vec[i] * stats.gamma.cdf(q, a=alpha, scale=beta_vec[i])

    return cdf_value
