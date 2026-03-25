import numpy as np
import pandas as pd
import pytest
from skbase.lookup import all_objects

from pgmpy.ci_tests import ChiSquare, FisherZ, IndependenceMatch, Pearsonr, _BaseCITest, get_ci_test


def test_ci_registry():
    all_tests = [
        ci_test.get_class_tag("name")
        for ci_test in all_objects(
            object_types=_BaseCITest,
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
