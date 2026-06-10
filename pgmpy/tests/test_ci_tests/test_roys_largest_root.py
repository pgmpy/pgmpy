import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.cross_decomposition import CCA

from pgmpy.ci_tests import RoysLargestRoot
from pgmpy.tests.test_ci_tests import _multivariate_fixtures

pillai_data = _multivariate_fixtures.pillai_data
skip_gh_actions = _multivariate_fixtures.skip_gh_actions


@skip_gh_actions
def test_roys_no_cond(pillai_data):
    expected_stats = [0.1572, 0.1572, 0.1359, 0.1000, 0.1359]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", [])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


@skip_gh_actions
def test_roys_indep(pillai_data):
    expected_stats = [0.0016, 0.0007, 0.0044, 0.0053, 0.0044]
    expected_pvalues = [0.2125, 0.4154, 0.1118, 0.0715, 0.1118]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["indep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


@skip_gh_actions
def test_roys_dependent(pillai_data):
    expected_stats = [0.1700, 0.2159, 0.1336, 0.1008, 0.1336]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_stats, computed_pvalues = [], []
    for df in pillai_data["dep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_stats.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_stats, expected_stats, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


def test_effect_size(pillai_data):
    expected_indep = [0.0026, 0.0004, 0.0003, 0.0023, 0.0003]
    expected_dep = [0.1698, 0.2181, 0.1328, 0.1008, 0.1328]

    for df, expected in zip(pillai_data["indep"], expected_indep):
        test = RoysLargestRoot(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.effect_size_ == pytest.approx(test.statistic_, abs=1e-2)
        assert test.effect_size_ == pytest.approx(expected, abs=1e-2)

    for df, expected in zip(pillai_data["dep"], expected_dep):
        test = RoysLargestRoot(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.effect_size_ == pytest.approx(test.statistic_, abs=1e-2)
        assert test.effect_size_ == pytest.approx(expected, abs=1e-2)


def test_roys_approx(pillai_data):
    for df in pillai_data["indep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", [])
        assert test.statistic_ >= 0.05
        assert test.p_value_ <= 0.05

    for df in pillai_data["indep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.statistic_ <= 0.1
        assert test.p_value_ >= 0.05

    for df in pillai_data["dep"]:
        test = RoysLargestRoot(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        assert test.statistic_ >= 0.05
        assert test.p_value_ <= 0.05


def test_roys_matches_muller_peterson_eq_28_for_p_not_equal_q():
    rng = np.random.default_rng(123)
    n_samples = 120

    z = rng.normal(size=n_samples)
    latent_x = 0.8 * z + rng.normal(size=n_samples)
    x = pd.qcut(latent_x, 4, labels=["x1", "x2", "x3", "x4"])
    x_num = pd.Series(x).cat.codes.to_numpy() - 1.5
    y = 0.5 * z + 0.1 * x_num + rng.normal(size=n_samples)
    data = pd.DataFrame({"X": x, "Y": y, "Z": z})

    test = RoysLargestRoot(data=data)
    test.run_test("X", "Y", ["Z"])

    res_x, _ = test.get_residuals("X", ["Z"])
    res_y, _ = test.get_residuals("Y", ["Z"])

    if isinstance(res_x, pd.Series):
        res_x = res_x.to_frame()
    if isinstance(res_y, pd.Series):
        res_y = res_y.to_frame()

    p, q = res_x.shape[1], res_y.shape[1]
    assert (p, q) == (3, 1)

    cca = CCA(scale=False, n_components=min(p, q))
    res_x_c, res_y_c = cca.fit_transform(res_x, res_y)
    cancor2 = np.array([np.corrcoef(res_x_c[:, i], res_y_c[:, i])[0, 1] ** 2 for i in range(min(p, q))])

    expected_statistic = float(np.max(cancor2))
    expected_pvalue = float(
        1.0
        - stats.f.cdf(
            (expected_statistic / p) / ((1.0 - expected_statistic) / (n_samples - p - 1)),
            p,
            n_samples - p - 1,
        )
    )

    assert test.statistic_ == pytest.approx(expected_statistic, abs=1e-2)
    assert test.p_value_ == pytest.approx(expected_pvalue, abs=1e-2)
