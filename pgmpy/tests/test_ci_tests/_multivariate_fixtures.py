import os

import pandas as pd
import pytest

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork

skip_gh_actions = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping residual tests on GitHub Actions.",
)


def _simulate_data(dependent: bool) -> pd.DataFrame:
    edges = [
        ("Z1", "X"),
        ("Z2", "X"),
        ("Z3", "X"),
        ("Z1", "Y"),
        ("Z2", "Y"),
        ("Z3", "Y"),
    ]
    if dependent:
        edges.append(("X", "Y"))

    model = LinearGaussianBayesianNetwork(edges)

    cpd_z1 = LinearGaussianCPD("Z1", [0], 1)
    cpd_z2 = LinearGaussianCPD("Z2", [0], 1)
    cpd_z3 = LinearGaussianCPD("Z3", [0], 1)
    cpd_x = LinearGaussianCPD("X", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
    cpd_y = (
        LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"])
        if dependent
        else LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
    )
    model.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y)

    return model.simulate(n_samples=1000, seed=42)


def _make_variants(df: pd.DataFrame) -> list[pd.DataFrame]:
    df_cont_cont = df.copy()
    df_cont_cont["Z2"] = pd.cut(
        df_cont_cont["Z2"],
        bins=4,
        ordered=False,
        labels=["z21", "z22", "z23", "z24"],
    )

    df_cat_cont = df_cont_cont.copy()
    df_cat_cont["X"] = pd.cut(
        df_cat_cont["X"],
        bins=3,
        ordered=False,
        labels=["x1", "x2", "x3"],
    )

    df_cat_cat = df_cont_cont.copy()
    df_cat_cat["X"] = pd.cut(
        df_cat_cat["X"],
        bins=3,
        ordered=False,
        labels=["x1", "x2", "x3"],
    )
    df_cat_cat["Y"] = pd.cut(
        df_cat_cat["Y"],
        bins=3,
        ordered=False,
        labels=["y1", "y2", "y3"],
    )

    df_ord_cont = df_cont_cont.copy()
    df_ord_cont["X"] = pd.cut(df_ord_cont["X"], bins=3)

    return [df.copy(), df_cont_cont, df_cat_cont, df_cat_cat, df_ord_cont]


@pytest.fixture
def pillai_data():
    return {
        "indep": _make_variants(_simulate_data(dependent=False)),
        "dep": _make_variants(_simulate_data(dependent=True)),
    }
