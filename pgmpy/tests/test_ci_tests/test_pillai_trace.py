import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from pgmpy.ci_tests import (
    GeneralizedCov,
    HotellingLawley,
    PillaiTrace,
    RoysLargestRoot,
    WilksLambda,
)
from pgmpy.tests.test_ci_tests import _multivariate_fixtures
from pgmpy.utils import CovarianceRegression

pillai_data = _multivariate_fixtures.pillai_data
skip_gh_actions = _multivariate_fixtures.skip_gh_actions


@skip_gh_actions
def test_pillai_no_cond(pillai_data):
    expected_coefs = [0.2015, 0.1665, 0.1184, 0.1049, 0.1184]
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
    expected_coefs = [0.0, 0.0, 0.0002, 0.0006, 0.0002]
    expected_pvalues = [0.9155, 0.9246, 0.8909, 0.9594, 0.8909]

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
    expected_coefs = [0.1950, 0.1948, 0.1077, 0.1373, 0.1077]
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
    expected_indep = [0.0, 0.0, 0.0002, 0.0003, 0.0002]
    expected_dep = [0.1950, 0.1948, 0.1077, 0.0686, 0.1077]

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


def test_pillai_bool_is_categorical_in_mixed_data():
    rng = np.random.default_rng(seed=4)
    data = pd.DataFrame({"A": rng.normal(size=80), "C": rng.normal(size=80), "G": rng.random(80) < 0.5})
    data["A"] += data["G"]
    data["H"] = rng.choice(["x", "y"], size=80)

    bool_test, category_test = PillaiTrace(data=data), PillaiTrace(data=data.assign(G=data["G"].astype("category")))
    bool_test("G", "A", ["C"])
    category_test("G", "A", ["C"])
    assert bool_test.p_value_ == pytest.approx(category_test.p_value_, rel=1e-9)


def test_default_estimator_types():
    rng = np.random.default_rng(0)
    n = 1000
    Z = rng.normal(size=n)
    X_cont = Z * 0.5 + rng.normal(size=n)
    Y_cont = Z * 0.5 + rng.normal(size=n)
    X_cat = pd.cut(X_cont, bins=4, labels=["a", "b", "c", "d"])

    # Continuous variables: estimator should be CovarianceRegression.
    test = PillaiTrace(data=pd.DataFrame({"X": X_cont, "Y": Y_cont, "Z": Z}))
    test("X", "Y", ["Z"])
    assert isinstance(test.estimator_x_, CovarianceRegression)
    assert isinstance(test.estimator_y_, CovarianceRegression)

    # Categorical X: estimator_x_ should be LogisticRegression, estimator_y_ CovarianceRegression.
    test = PillaiTrace(data=pd.DataFrame({"X": X_cat, "Y": Y_cont, "Z": Z}))
    test("X", "Y", ["Z"])
    assert isinstance(test.estimator_x_, LogisticRegression)
    assert isinstance(test.estimator_y_, CovarianceRegression)


def test_linearity_tag():
    rng = np.random.default_rng(0)
    n = 200
    data = pd.DataFrame({"X": rng.normal(size=n), "Y": rng.normal(size=n), "Z": rng.normal(size=n)})

    assert PillaiTrace.get_class_tag("assumption:linearity") is False
    assert PillaiTrace(data=data).get_tag("assumption:linearity") is True
    assert (
        PillaiTrace(data=data, estimator=RandomForestRegressor(random_state=0)).get_tag("assumption:linearity") is False
    )


def test_custom_discrete_estimator():
    rng = np.random.default_rng(0)
    n = 200
    Z = rng.normal(size=n)
    X = pd.cut(Z * 0.5 + rng.normal(size=n), bins=3, labels=["a", "b", "c"])
    Y = pd.cut(Z * 0.5 + rng.normal(size=n), bins=3, labels=["x", "y", "z"])
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    test = PillaiTrace(data=df, estimator=RandomForestClassifier(random_state=0))
    test("X", "Y", ["Z"])
    assert isinstance(test.estimator_x_, RandomForestClassifier)
    assert isinstance(test.estimator_y_, RandomForestClassifier)


@pytest.mark.parametrize(
    "ci_test_cls",
    [PillaiTrace, HotellingLawley, RoysLargestRoot, WilksLambda, GeneralizedCov],
)
def test_constant_categorical_target(ci_test_cls):
    rng = np.random.default_rng(0)
    n = 100
    data = pd.DataFrame(
        {
            "X": pd.Categorical(["only_category"] * n),
            "Y": rng.normal(size=n),
            "Z": rng.normal(size=n),
        }
    )
    test = ci_test_cls(data=data)
    # Categorical variable with single unique category should not raise ValueError and return independent
    assert test.is_independent("X", "Y", ["Z"]) is True
    assert test.p_value_ == 1.0
    assert test.estimator_x_ is None


def test_conditioning_dummy_column_name_matches_target():
    rng = np.random.default_rng(0)
    n = 100
    # Conditioning variable A has category '1', so get_dummies produces dummy column 'A_1'
    a_cat = pd.Categorical(["1" if i % 2 == 0 else "2" for i in range(n)])
    x_cont = rng.normal(size=n)
    a_1_cont = rng.normal(size=n)
    data = pd.DataFrame({"X": x_cont, "A_1": a_1_cont, "A": a_cat})

    test = PillaiTrace(data=data)
    # Target 'A_1' name collides with dummy column 'A_1' generated from conditioning variable 'A'
    result = test("X", "A_1", ["A"])
    assert isinstance(result, (bool, np.bool_))
    assert isinstance(test.estimator_y_, CovarianceRegression)
