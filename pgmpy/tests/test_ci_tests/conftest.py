import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def test_data():
    rng = np.random.default_rng(seed=42)

    # Continuous Independent
    df_ind = pd.DataFrame(rng.standard_normal(size=(10000, 3)), columns=["X", "Y", "Z"])

    # Continuous Conditionally Independent
    Z = rng.normal(size=10000)
    X = 3 * Z + rng.normal(loc=0, scale=0.1, size=10000)
    Y = 2 * Z + rng.normal(loc=0, scale=0.1, size=10000)
    df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    # Continuous Conditionally Independent with Multiple Parents
    Z1 = rng.normal(size=10000)
    Z2 = rng.normal(size=10000)
    X_mul = 3 * Z1 + 2 * Z2 + rng.normal(loc=0, scale=0.1, size=10000)
    Y_mul = 2 * Z1 + 3 * Z2 + rng.normal(loc=0, scale=0.1, size=10000)
    df_cind_mul = pd.DataFrame({"X": X_mul, "Y": Y_mul, "Z1": Z1, "Z2": Z2})

    # Continuous V-structure
    X_v = rng.normal(size=10000)
    Y_v = rng.normal(size=10000)
    Z_v = 2 * X_v + 2 * Y_v + rng.normal(loc=0, scale=0.1, size=10000)
    df_vstruct = pd.DataFrame({"X": X_v, "Y": Y_v, "Z": Z_v})

    # Discrete Independent
    df_disc_ind = pd.DataFrame(
        rng.integers(0, 3, size=(10000, 3)), columns=["X", "Y", "Z"]
    ).astype("category")

    # Discrete Conditionally Independent
    conditional_probs = {0: [0.7, 0.2, 0.1], 1: [0.2, 0.5, 0.3], 2: [0.1, 0.3, 0.6]}
    Z_disc = rng.integers(0, 3, size=10000)
    X_disc = np.array([rng.choice(3, p=conditional_probs[z]) for z in Z_disc])
    Y_disc = np.array([rng.choice(3, p=conditional_probs[z]) for z in Z_disc])
    df_disc_cind = pd.DataFrame({"X": X_disc, "Y": Y_disc, "Z": Z_disc}).astype(
        "category"
    )

    # Discrete Conditionally Independent with Multiple Parents
    def discrete_mean(z1, z2):
        return np.round(0.5 * z1 + 0.5 * z2).astype(int)

    Z1_disc = rng.integers(0, 3, size=10000)
    Z2_disc = rng.integers(0, 3, size=10000)
    X_disc_mul = np.array(
        [
            rng.choice(3, p=conditional_probs[discrete_mean(z1, z2)])
            for z1, z2 in zip(Z1_disc, Z2_disc)
        ]
    )
    Y_disc_mul = np.array(
        [
            rng.choice(3, p=conditional_probs[discrete_mean(z1, z2)])
            for z1, z2 in zip(Z1_disc, Z2_disc)
        ]
    )
    df_disc_cind_mul = pd.DataFrame(
        {"X": X_disc_mul, "Y": Y_disc_mul, "Z1": Z1_disc, "Z2": Z2_disc}
    ).astype("category")

    # Discrete V-structure
    X_disc_v = rng.integers(0, 3, size=10000)
    Y_disc_v = rng.integers(0, 3, size=10000)
    Z_disc_v = np.array(
        [
            rng.choice(3, p=conditional_probs[discrete_mean(x, y)])
            for x, y in zip(X_disc_v, Y_disc_v)
        ]
    )
    df_disc_vstruct = pd.DataFrame(
        {"X": X_disc_v, "Y": Y_disc_v, "Z": Z_disc_v}
    ).astype("category")

    return {
        "cont_ind": df_ind,
        "cont_cind": df_cind,
        "cont_cind_mul": df_cind_mul,
        "cont_vstruct": df_vstruct,
        "disc_ind": df_disc_ind,
        "disc_cind": df_disc_cind,
        "disc_cind_mul": df_disc_cind_mul,
        "disc_vstruct": df_disc_vstruct,
    }
