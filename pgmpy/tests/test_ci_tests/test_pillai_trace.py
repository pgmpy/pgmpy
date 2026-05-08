import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pgmpy.ci_tests import PillaiTrace
from pgmpy.tests.test_ci_tests import _multivariate_fixtures

pillai_data = _multivariate_fixtures.pillai_data
skip_gh_actions = _multivariate_fixtures.skip_gh_actions


@skip_gh_actions
def test_pillai_no_cond(pillai_data):
    expected_coefs = [0.1572, 0.1572, 0.1359, 0.1068, 0.1359]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", [])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_coefs, expected_coefs, atol=1e-2)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-2)


@skip_gh_actions
def test_pillai_indep(pillai_data):
    expected_coefs = [0.0016, 0.0007, 0.0044, 0.0055, 0.0044]
    expected_pvalues = [0.2125, 0.4154, 0.1118, 0.2406, 0.1118]

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_coefs, expected_coefs, atol=1e-2)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-2)


@skip_gh_actions
def test_pillai_dependent(pillai_data):
    expected_coefs = [0.1698, 0.2181, 0.1328, 0.1595, 0.1328]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["dep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_coefs, expected_coefs, atol=1e-2)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-2)


def test_pillai_tests_approx(pillai_data):
    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", [])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) >= 0.1)
    assert np.all(np.array(computed_pvalues) <= 0.05)

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) <= 0.1)
    assert np.all(np.array(computed_pvalues)[:3] >= 0.05)
    assert np.all(np.array(computed_pvalues)[4:] >= 0.05)

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["dep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) >= 0.05)
    assert np.all(np.array(computed_pvalues) <= 0.05)


def test_effect_size(pillai_data):
    expected_indep = [0.0026, 0.0004, 0.0003, 0.0013, 0.0003]
    expected_dep = [0.1698, 0.2181, 0.1328, 0.0798, 0.1328]

    for df, expected in zip(pillai_data["indep"], expected_indep):
        test = PillaiTrace(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.effect_size_ == pytest.approx(expected, abs=1e-2)

    for df, expected in zip(pillai_data["dep"], expected_dep):
        test = PillaiTrace(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.effect_size_ == pytest.approx(expected, abs=1e-2)


def test_pillai_approx_discrete():
    # Tests approximate behavior with actual categorical variables (string labels) so that
    # the categorical residual path in get_residuals is exercised on GitHub CI.
    rng = np.random.default_rng(0)
    n = 1000
    Z = rng.normal(size=n)
    X_cont = Z * 0.5 + rng.normal(size=n)
    Y_cont_indep = Z * 0.5 + rng.normal(size=n)
    Y_cont_dep = Z * 0.5 + X_cont * 1.0 + rng.normal(size=n)

    X_cat = pd.cut(X_cont, bins=4, labels=["a", "b", "c", "d"])
    Y_cat_indep = pd.cut(Y_cont_indep, bins=4, labels=["p", "q", "r", "s"])
    Y_cat_dep = pd.cut(Y_cont_dep, bins=4, labels=["p", "q", "r", "s"])

    # cat-cont: categorical X, continuous Y
    test = PillaiTrace(data=pd.DataFrame({"X": X_cat, "Y": Y_cont_indep, "Z": Z}))
    test("X", "Y", ["Z"])
    assert test.p_value_ > 0.05

    test = PillaiTrace(data=pd.DataFrame({"X": X_cat, "Y": Y_cont_dep, "Z": Z}))
    test("X", "Y", ["Z"])
    assert test.p_value_ < 0.05

    # cat-cat: both X and Y categorical
    test = PillaiTrace(data=pd.DataFrame({"X": X_cat, "Y": Y_cat_indep, "Z": Z}))
    test("X", "Y", ["Z"])
    assert test.p_value_ > 0.05

    test = PillaiTrace(data=pd.DataFrame({"X": X_cat, "Y": Y_cat_dep, "Z": Z}))
    test("X", "Y", ["Z"])
    assert test.p_value_ < 0.05


def test_pillai_linear_regression_estimator():
    rng = np.random.default_rng(0)
    n = 2000
    Z = rng.normal(size=(n, 2))
    X = Z @ [0.5, 0.5] + rng.normal(size=n)
    Y_indep = Z @ [0.5, 0.5] + rng.normal(size=n)
    Y_dep = Z @ [0.5, 0.5] + 0.8 * X + rng.normal(size=n)
    df_indep = pd.DataFrame({"X": X, "Y": Y_indep, "Z1": Z[:, 0], "Z2": Z[:, 1]})
    df_dep = pd.DataFrame({"X": X, "Y": Y_dep, "Z1": Z[:, 0], "Z2": Z[:, 1]})

    test_indep = PillaiTrace(data=df_indep, estimator=LinearRegression())
    result_indep = test_indep("X", "Y", ["Z1", "Z2"])
    assert isinstance(result_indep, (bool, np.bool_))
    assert test_indep.p_value_ > 0.05

    test_dep = PillaiTrace(data=df_dep, estimator=LinearRegression())
    result_dep = test_dep("X", "Y", ["Z1", "Z2"])
    assert isinstance(result_dep, (bool, np.bool_))
    assert test_dep.p_value_ < 0.05


def test_pillai_linear_regression_no_predict_proba():
    rng = np.random.default_rng(0)
    n = 200
    Z = rng.normal(size=n)
    X = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    Y = rng.normal(size=n)
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    test = PillaiTrace(data=df, estimator=LinearRegression())
    with pytest.raises(ValueError, match="predict_proba"):
        test("X", "Y", ["Z"])
