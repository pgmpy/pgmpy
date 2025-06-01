import numpy as np
import pandas as pd
from scipy.stats import bernoulli, expon, multinomial, norm, uniform


def linear_gaussian(
    n_samples=1000,
    effect_size=1.0,
    noise_std=1.0,
    n_cond_vars=1,
    seed=None,
):
    """
    Linear Gaussian DGP:
        Z ~ N(0, 1)
        X = a1 * Z + e1
        Y = a2 * Z + e2
        e1, e2 ~ N(0, noise_std^2)
    Parameters:
        n_samples: Number of samples
        effect_size: Coefficient for Z in X and Y
        noise_std: Standard deviation of noise
        n_cond_vars: Number of variables in Z (vector-valued Z)
        seed: Random seed
    Returns:
        DataFrame with columns ['X', 'Y', 'Z1', ...]
    Reference: Spirtes, Glymour & Scheines (2000), "Causation, Prediction, and Search"
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_samples, n_cond_vars))
    e1 = rng.normal(scale=noise_std, size=n_samples)
    e2 = rng.normal(scale=noise_std, size=n_samples)
    X = effect_size * Z.sum(axis=1) + e1
    Y = effect_size * Z.sum(axis=1) + e2
    data = {"X": X, "Y": Y}
    for j in range(n_cond_vars):
        data[f"Z{j+1}"] = Z[:, j]
    return pd.DataFrame(data)


def nonlinear_gaussian(n_samples=1000, effect_size=1.0, noise_std=1.0, seed=None):
    """
    Nonlinear Gaussian DGP:
        Z ~ N(0, 1)
        X = sin(a1 * Z) + e1
        Y = exp(a2 * Z) + e2
        e1, e2 ~ N(0, noise_std^2)
    Parameters:
        n_samples: Number of samples
        effect_size: Amplitude for Z in X and Y
        noise_std: Standard deviation of noise
        seed: Random seed
    Returns:
        DataFrame with columns ['X', 'Y', 'Z']
    Reference: Peters et al (2011), "Causal inference by using invariant prediction"
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n_samples)
    e1 = rng.normal(scale=noise_std, size=n_samples)
    e2 = rng.normal(scale=noise_std, size=n_samples)
    X = np.sin(effect_size * Z) + e1
    Y = np.exp(effect_size * Z * 0.2) + e2  # Scaled for numerical stability
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


# --- NEW DGMs ADDED LATER ---


def discrete_categorical(
    n_samples=1000, n_categories=3, effect_size=1.0, noise_prob=0.05, seed=None
):
    """
    Discrete (categorical) DGP:
        Z ~ Multinomial(n_categories)
        X = Z + noise
        Y = Z + noise
    Parameters:
        n_samples: Number of samples
        n_categories: Number of categories
        effect_size: Not used, for API consistency
        noise_prob: Probability to perturb X or Y
        seed: Random seed
    Returns:
        DataFrame with columns ['X', 'Y', 'Z']
    Reference: Scutari, Denis (2021), "Bayesian Networks: With Examples in R"
    """
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, n_categories, size=n_samples)
    X = Z.copy()
    Y = Z.copy()
    for arr in (X, Y):
        flips = rng.random(n_samples) < noise_prob
        arr[flips] = rng.integers(0, n_categories, size=flips.sum())
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def mixed_data(n_samples=1000, n_cat=2, effect_size=1.0, noise_std=1.0, seed=None):
    """
    Mixed continuous and categorical DGP:
        Z (categorical)
        X = a * Z + e1
        Y = b * Z + e2
    Parameters:
        n_samples: Number of samples
        n_cat: Number of categories for Z
        effect_size: Effect size for Z on X and Y
        noise_std: Std for additive noise
        seed: Random seed
    Returns:
        DataFrame with columns ['X', 'Y', 'Z']
    Reference: Ghassami et al (2017), "Learning Mixed Graphical Models"
    """
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, n_cat, size=n_samples)
    e1 = rng.normal(scale=noise_std, size=n_samples)
    e2 = rng.normal(scale=noise_std, size=n_samples)
    X = effect_size * Z + e1
    Y = 0.5 * effect_size * Z + e2
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def non_gaussian_continuous(n_samples=1000, effect_size=1.0, noise_std=1.0, seed=None):
    """
    Non-Gaussian continuous DGP:
      Z ~ Uniform(-2, 2)
      X = |Z| + e1, e1 ~ Exp(1)
      Y = Z^2 + e2,  e2 ~ Exp(1)
    Parameters:
        n_samples: Number of samples
        effect_size: Scales the effect of Z
        noise_std: Not used (kept for API consistency)
        seed: Random seed
    Returns:
        DataFrame with columns ['X', 'Y', 'Z']
    Reference: Shimizu et al (2006), "A Linear Non-Gaussian Acyclic Model for Causal Discovery"
    """
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-2, 2, size=n_samples)
    e1 = rng.exponential(scale=1.0, size=n_samples)
    e2 = rng.exponential(scale=1.0, size=n_samples)
    X = effect_size * np.abs(Z) + e1
    Y = effect_size * Z**2 + e2
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


#  NEW DGMS WILL BE ADDED HERE


def user_defined(generator_func, **kwargs):
    """
    Allows users to plug in a custom DGP function.
    The function must return a pandas DataFrame.
    """
    return generator_func(**kwargs)


# Optionally, a registry for easy reference
DGP_REGISTRY = {
    "linear_gaussian": linear_gaussian,
    "nonlinear_gaussian": nonlinear_gaussian,
    "discrete_categorical": discrete_categorical,
    "mixed_data": mixed_data,
    "non_gaussian_continuous": non_gaussian_continuous,
    "user_defined": user_defined,
}

# Example usage for testing:
if __name__ == "__main__":
    for name, func in DGP_REGISTRY.items():
        if name == "user_defined":
            continue
        print(f"Generating data for {name}...")
        df = func(n_samples=100, seed=42)
        print(df.head())
