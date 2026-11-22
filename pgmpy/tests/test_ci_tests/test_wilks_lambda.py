import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from pgmpy.ci_tests import WilksLambda
from pgmpy.tests.test_ci_tests import _multivariate_fixtures
from pgmpy.utils import CovarianceRegression

pillai_data = _multivariate_fixtures.pillai_data
skip_gh_actions = _multivariate_fixtures.skip_gh_actions


@skip_gh_actions
def test_wilks_no_cond(pillai_data):
    expected_stats = [0.7985, 0.8335, 0.8816, 0.8962, 0.8816]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = WilksLambda(data=df)
        test("X", "Y", [])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-2)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-2)


@skip_gh_actions
def test_wilks_indep(pillai_data):
    expected_stats = [1.0, 1.0, 0.9998, 0.9994, 0.9998]
    expected_pvalues = [0.9155, 0.9246, 0.8909, 0.9595, 0.8909]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-2)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-2)


@skip_gh_actions
def test_wilks_dependent(pillai_data):
    expected_stats = [0.8050, 0.8052, 0.8923, 0.8670, 0.8923]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["dep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-2)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-2)


def test_effect_size(pillai_data):
    expected_indep = [0.0, 0.0, 0.0002, 0.0003, 0.0002]
    expected_dep = [0.1950, 0.1948, 0.1077, 0.0688, 0.1077]

    for df, expected in zip(pillai_data["indep"], expected_indep):
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.effect_size_ == pytest.approx(expected, abs=1e-2)

    for df, expected in zip(pillai_data["dep"], expected_dep):
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.effect_size_ == pytest.approx(expected, abs=1e-2)


def test_wilks_approx(pillai_data):
    for df in pillai_data["indep"]:
        test = WilksLambda(data=df)
        test("X", "Y", [])
        assert test.statistic_ < 1.0
        assert test.p_value_ <= 0.05

    for df in pillai_data["indep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.statistic_ > 0.9
        assert test.p_value_ >= 0.05

    for df in pillai_data["dep"]:
        test = WilksLambda(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.statistic_ < 0.95
        assert test.p_value_ <= 0.05


def test_default_estimator_types():
    rng = np.random.default_rng(0)
    n = 1000
    Z = rng.normal(size=n)
    X_cont = Z * 0.5 + rng.normal(size=n)
    Y_cont = Z * 0.5 + rng.normal(size=n)
    X_cat = pd.cut(X_cont, bins=4, labels=["a", "b", "c", "d"])

    # Continuous variables: estimator should be CovarianceRegression.
    test = WilksLambda(data=pd.DataFrame({"X": X_cont, "Y": Y_cont, "Z": Z}))
    test("X", "Y", ["Z"])
    assert isinstance(test.estimator_x_, CovarianceRegression)
    assert isinstance(test.estimator_y_, CovarianceRegression)

    # Categorical X: estimator_x_ should be LogisticRegression, estimator_y_ CovarianceRegression.
    test = WilksLambda(data=pd.DataFrame({"X": X_cat, "Y": Y_cont, "Z": Z}))
    test("X", "Y", ["Z"])
    assert isinstance(test.estimator_x_, LogisticRegression)
    assert isinstance(test.estimator_y_, CovarianceRegression)


def test_linearity_tag():
    rng = np.random.default_rng(0)
    n = 200
    data = pd.DataFrame({"X": rng.normal(size=n), "Y": rng.normal(size=n), "Z": rng.normal(size=n)})

    assert WilksLambda.get_class_tag("assumption:linearity") is False
    assert WilksLambda(data=data).get_tag("assumption:linearity") is True
    assert (
        WilksLambda(data=data, estimator=RandomForestRegressor(random_state=0)).get_tag("assumption:linearity") is False
    )
