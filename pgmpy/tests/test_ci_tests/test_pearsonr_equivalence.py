import os

import numpy as np
import pytest

from pgmpy.ci_tests import PearsonrEquivalence
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


@pytest.fixture
def pearson_equivalence_data():
    np.random.seed(42)

    model_dep = LinearGaussianBayesianNetwork(
        [
            ("Z1", "X"),
            ("Z2", "X"),
            ("Z3", "X"),
            ("Z1", "Y"),
            ("Z2", "Y"),
            ("Z3", "Y"),
            ("X", "Y"),
        ]
    )
    cpd_z1 = LinearGaussianCPD("Z1", [0], 1)
    cpd_z2 = LinearGaussianCPD("Z2", [0], 1)
    cpd_z3 = LinearGaussianCPD("Z3", [0], 1)
    cpd_x = LinearGaussianCPD("X", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
    cpd_y_dep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"])
    model_dep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_dep)
    df_dep = model_dep.simulate(n_samples=1000, seed=42)

    return df_dep


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping residual tests on GitHub Actions.",
)
def test_pearsonr_equivalence(pearson_equivalence_data):
    test = PearsonrEquivalence(data=pearson_equivalence_data, delta_threshold=0.1)

    assert not test("X", "Y", ["Z1", "Z2", "Z3"], significance_level=0.05)

    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert round(test.statistic_, 2) == pytest.approx(0.43, abs=1e-2)
    assert round(test.p_value_, 2) == pytest.approx(1.0, abs=1e-2)


def test_effect_size(pearson_equivalence_data):
    test = PearsonrEquivalence(data=pearson_equivalence_data, delta_threshold=0.1)
    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.effect_size_ == pytest.approx(0.4056, abs=1e-2)


def test_pearsonr_equivalence_approx(pearson_equivalence_data):
    test = PearsonrEquivalence(data=pearson_equivalence_data, delta_threshold=0.1)

    assert not test("X", "Y", ["Z1", "Z2", "Z3"], significance_level=0.05)

    test("X", "Y", ["Z1", "Z2", "Z3"])
    assert test.statistic_ <= 0.7
    assert test.p_value_ <= 1.0
