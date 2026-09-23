import os

import numpy as np
import pandas as pd
import pytest

MULTIVARIATE_FIXTURE_SEED = 42

skip_gh_actions = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping residual tests on GitHub Actions.",
)


def _simulate_data(dependent: bool, z2_categorical: bool, seed: int = MULTIVARIATE_FIXTURE_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z1, z2, z3, noise_x, noise_y = rng.standard_normal((5, 1000))

    z2_column = z2
    if z2_categorical:
        z2_column = pd.cut(
            z2,
            bins=4,
            ordered=False,
            labels=["z21", "z22", "z23", "z24"],
        )
        z2 = z2_column.codes.astype(float)

    x = 0.5 * (z1 + z2 + z3) + noise_x
    y = 0.5 * (z1 + z2 + z3) + noise_y
    if dependent:
        y += 0.5 * x

    return pd.DataFrame({"Z1": z1, "Z2": z2_column, "Z3": z3, "X": x, "Y": y})


def _make_variants(df: pd.DataFrame, df_z2_categorical: pd.DataFrame) -> list[pd.DataFrame]:
    df_cat_cont = df_z2_categorical.copy()
    df_cat_cont["X"] = pd.cut(
        df_cat_cont["X"],
        bins=3,
        ordered=False,
        labels=["x1", "x2", "x3"],
    )

    df_cat_cat = df_cat_cont.copy()
    df_cat_cat["Y"] = pd.cut(
        df_cat_cat["Y"],
        bins=3,
        ordered=False,
        labels=["y1", "y2", "y3"],
    )

    df_ord_cont = df_z2_categorical.copy()
    df_ord_cont["X"] = pd.cut(df_ord_cont["X"], bins=3)

    return [df, df_z2_categorical, df_cat_cont, df_cat_cat, df_ord_cont]


def _build_pillai_data(seed: int = MULTIVARIATE_FIXTURE_SEED) -> dict[str, list[pd.DataFrame]]:
    return {
        "indep": _make_variants(
            _simulate_data(dependent=False, z2_categorical=False, seed=seed),
            _simulate_data(dependent=False, z2_categorical=True, seed=seed),
        ),
        "dep": _make_variants(
            _simulate_data(dependent=True, z2_categorical=False, seed=seed),
            _simulate_data(dependent=True, z2_categorical=True, seed=seed),
        ),
    }


@pytest.fixture
def pillai_data():
    return _build_pillai_data()
