import numpy as np
import pandas as pd
import pytest
from skbase.lookup import all_objects
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

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


RESIDUAL_TESTS = [GCM, PillaiTrace, GeneralizedCov, WilksLambda, HotellingLawley, RoysLargestRoot]


@pytest.mark.parametrize("cls", RESIDUAL_TESTS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        (LinearRegression(), True),
        (Ridge(), True),
        (RandomForestRegressor(random_state=0), False),
        (KNeighborsRegressor(), None),
    ],
    ids=["linear_model", "ridge", "random_forest", "other_estimator"],
)
def test_linearity_assumption_tag_follows_estimator(cont_data, cls, estimator, expected):
    assert cls.get_class_tag("assumption:linearity") is False
    assert cls(data=cont_data, estimator=estimator).get_tag("assumption:linearity") is expected


@pytest.mark.parametrize(
    "cls", [PillaiTrace, GeneralizedCov, WilksLambda, HotellingLawley, RoysLargestRoot], ids=lambda cls: cls.__name__
)
def test_linearity_assumption_tag_default_estimator(cont_data, cls):
    assert cls(data=cont_data).get_tag("assumption:linearity") is False


def test_linearity_assumption_tag_gcm_default_estimator(cont_data):
    assert GCM(data=cont_data).get_tag("assumption:linearity") is True


def test_requires_data_none_treated_as_required(monkeypatch, cont_data):
    monkeypatch.setitem(ChiSquare._tags, "requires_data", None)
    with pytest.raises(ValueError, match="requires data"):
        get_ci_test(test="chi_square", data=None)
    assert isinstance(get_ci_test(test="chi_square", data=cont_data), ChiSquare)
