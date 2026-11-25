import numpy as np
import pandas as pd
import pytest
from skbase.lookup import all_objects
from sklearn.linear_model import LogisticRegression

from pgmpy.ci_tests import (
    GCM,
    BaseCITest,
    ChiSquare,
    FisherZ,
    GeneralizedCov,
    HotellingLawley,
    IndependenceMatch,
    Pearsonr,
    PillaiTrace,
    RoysLargestRoot,
    WilksLambda,
    get_ci_test,
)
from pgmpy.utils import CovarianceRegression


def test_ci_registry():
    all_tests = [
        ci_test.get_class_tag("name")
        for ci_test in all_objects(
            object_types=BaseCITest,
            package_name="pgmpy.ci_tests",
            return_names=False,
        )
    ]

    assert "chi_square" in all_tests
    assert "g_sq" in all_tests
    assert "log_likelihood" in all_tests
    assert "modified_log_likelihood" in all_tests
    assert "pearsonr" in all_tests
    assert "fisher_z" in all_tests
    assert "pillai" in all_tests
    assert "gcm" in all_tests
    assert "generalized_cov" in all_tests
    assert "wilks_lambda" in all_tests
    assert "roys_largest_root" in all_tests
    assert "hotelling_lawley" in all_tests


def test_filter_ci_tests_by_data_type():
    names = {
        cls.__name__
        for cls in all_objects(
            object_types=BaseCITest,
            package_name="pgmpy.ci_tests",
            return_names=False,
            filter_tags={"data_types": "continuous"},
        )
    }
    assert {"Pearsonr", "FisherZ", "PillaiTrace"} <= names
    assert "ChiSquare" not in names


@pytest.fixture
def cont_data():
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(rng.standard_normal((100, 3)), columns=["X", "Y", "Z"])


@pytest.fixture
def disc_data():
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame({"X": rng.choice(["a", "b"], 100), "Y": rng.choice(["c", "d"], 100)})


def test_pass_through_instance(cont_data):
    existing = Pearsonr(data=cont_data)
    assert get_ci_test(test=existing) is existing


def test_pass_through_callable():
    def my_test(X, Y, Z, significance_level=0.05):
        return True

    assert get_ci_test(test=my_test) is my_test


def test_by_name(disc_data):
    assert isinstance(get_ci_test(test="chi_square", data=disc_data), ChiSquare)


def test_by_name_case_insensitive(disc_data):
    assert isinstance(get_ci_test(test="Chi_Square", data=disc_data), ChiSquare)


def test_by_name_fisher_z(cont_data):
    assert isinstance(get_ci_test(test="fisher_z", data=cont_data), FisherZ)


def test_auto_detect_continuous(cont_data):
    assert isinstance(get_ci_test(data=cont_data), Pearsonr)


def test_auto_detect_discrete(disc_data):
    assert isinstance(get_ci_test(data=disc_data), ChiSquare)


def test_no_data_requires_data():
    with pytest.raises(ValueError):
        get_ci_test(test="chi_square", data=None)


def test_requires_data_false():
    assert isinstance(get_ci_test(test="independence_match"), IndependenceMatch)


def test_none_test_none_data():
    with pytest.raises(ValueError):
        get_ci_test(test=None, data=None)


def test_unknown_name(disc_data):
    with pytest.raises(ValueError):
        get_ci_test(test="nonexistent_test", data=disc_data)


def test_invalid_type():
    with pytest.raises(ValueError):
        get_ci_test(test=123)


@pytest.mark.parametrize(
    "cls",
    [GCM, PillaiTrace, GeneralizedCov, WilksLambda, HotellingLawley, RoysLargestRoot],
)
def test_residual_ci_tests_default_continuous_estimator(cls, cont_data):
    test = cls(data=cont_data)
    test("X", "Y", ["Z"])
    assert isinstance(test.estimator_x_, CovarianceRegression)
    assert isinstance(test.estimator_y_, CovarianceRegression)


@pytest.mark.parametrize(
    "cls",
    [PillaiTrace, GeneralizedCov, WilksLambda, HotellingLawley, RoysLargestRoot],
)
def test_residual_ci_tests_default_discrete_estimator(cls):
    rng = np.random.default_rng(seed=42)
    disc_data = pd.DataFrame(
        {
            "X": rng.choice(["a", "b"], 100),
            "Y": rng.choice(["c", "d"], 100),
            "Z": rng.choice(["e", "f"], 100),
        }
    )
    test = cls(data=disc_data)
    test("X", "Y", ["Z"])
    assert isinstance(test.estimator_x_, LogisticRegression)
    assert isinstance(test.estimator_y_, LogisticRegression)


def test_residual_ci_tests_degenerate_and_name_collision():
    rng = np.random.default_rng(seed=42)
    n = 60

    # Constant categorical target returns independence without raising
    df_const_cat = pd.DataFrame({"X": ["only"] * n, "Y": rng.choice(["a", "b"], n)})
    for cls in (PillaiTrace, GeneralizedCov):
        test = cls(data=df_const_cat, rng=42) if cls is GeneralizedCov else cls(data=df_const_cat)
        assert test("X", "Y", [])
        assert test.p_value_ == 1.0

    # Deterministic continuous target (zero residual variance) returns independence
    z = rng.normal(size=n)
    df_det = pd.DataFrame({"X": 2.0 * z + 1.0, "Y": rng.normal(size=n), "Z": z})
    for cls in (PillaiTrace, HotellingLawley, RoysLargestRoot, WilksLambda):
        test = cls(data=df_det)
        assert test("X", "Y", ["Z"])
        assert test.p_value_ == 1.0

    # Target named 'Z_a' does not collide with dummy column 'Z_a' from categorical Z
    df_collision = pd.DataFrame(
        {
            "Z_a": rng.normal(size=n),
            "Y": rng.normal(size=n),
            "Z": rng.choice(["a", "b"], size=n),
        }
    )
    test = PillaiTrace(data=df_collision)
    test("Z_a", "Y", ["Z"])
    assert 0.0 <= test.p_value_ <= 1.0
    u = np.tile([-1.0, 1.0], 50)
    v = np.tile([1.0, 1.0, -1.0, -1.0], 25)
    df_unit = pd.DataFrame({"X": u, "Y": u + 0.1 * v})
    df_scaled = pd.DataFrame({"X": 1e-10 * u, "Y": u + 0.1 * v})
    for cls in (PillaiTrace, HotellingLawley, RoysLargestRoot, WilksLambda):
        t_orig = cls(data=df_unit)
        t_scaled = cls(data=df_scaled)
        assert t_orig("X", "Y", []) == t_scaled("X", "Y", [])
        assert t_scaled.p_value_ < 0.05
