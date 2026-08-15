import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery import ANM


@pytest.fixture
def nonlinear_data(n=1000):
    """Generate additive-noise data with X -> Y, where Y = X ** 3 + noise."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, n)
    y = x**3 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"X": x, "Y": y})


def test_init():
    est = ANM()
    assert est.get_params() == {"regressor": None, "scoring_method": "independence"}


def test_score_algorithm_tags():
    from pgmpy.causal_discovery.bivariate_scores import EntropyScore, GaussScore, IndependenceScore

    scores = {
        "independence": IndependenceScore(),
        "entropy": EntropyScore(),
        "gauss": GaussScore(),
    }
    for name, score in scores.items():
        assert score.get_tag("name") == name
        assert score.get_tag("supported_algorithms") == ["anm"]


def test_fit_recovers_direction(nonlinear_data):
    data = nonlinear_data
    est = ANM().fit(data)

    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ == pytest.approx(0.00027103171629108734, rel=1e-4)
    assert est.backward_score_ == pytest.approx(0.002179257501560248, rel=1e-4)
    assert est.forward_score_ < est.backward_score_
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.adjacency_matrix_.loc["Y", "X"] == 0
    assert est.score(true_graph=est.causal_graph_, metric="SHD") == 0


def test_reproducible(nonlinear_data):
    est1 = ANM().fit(nonlinear_data)
    est2 = ANM().fit(nonlinear_data)
    assert est1.forward_score_ == est2.forward_score_
    assert est1.backward_score_ == est2.backward_score_


@pytest.mark.parametrize("score", ["independence", "entropy", "gauss"])
def test_builtin_scores_recover_direction(nonlinear_data, score):
    est = ANM(scoring_method=score).fit(nonlinear_data)
    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ < est.backward_score_


def test_score_instance_is_used(nonlinear_data):
    from pgmpy.causal_discovery.bivariate_scores import EntropyScore, IndependenceScore

    for score in (
        EntropyScore(method="vasicek"),
        IndependenceScore(ci_test="pearsonr"),
        IndependenceScore(criterion="p_value"),
    ):
        est = ANM(scoring_method=score).fit(nonlinear_data)
        assert list(est.causal_graph_.edges()) == [("X", "Y")]


def test_nan_or_tied_direction_scores_raise(nonlinear_data):
    from sklearn.linear_model import LinearRegression

    for score in (np.nan, 0.0):
        with pytest.raises(ValueError):
            ANM(regressor=LinearRegression(), scoring_method=lambda x, y: score).fit(nonlinear_data)


def test_ci_test_instance_uses_score_input():
    from pgmpy.causal_discovery.bivariate_scores import IndependenceScore
    from pgmpy.ci_tests import Pearsonr

    original_data = pd.DataFrame({"_x": [0, 1, 2, 3], "_y": [0, 1, 2, 3]})
    score = IndependenceScore(ci_test=Pearsonr(data=original_data))

    assert score([0, 1, 2, 3], [-1, 1, 1, -1]) == pytest.approx(0)


def test_score_hyperparameters_surface_errors(nonlinear_data):
    from pgmpy.causal_discovery.bivariate_scores import EntropyScore, IndependenceScore

    with pytest.raises(ValueError):  # scipy rejects an unknown differential_entropy method
        ANM(scoring_method=EntropyScore(method="not_a_method")).fit(nonlinear_data)
    with pytest.raises(ValueError):  # get_ci_test rejects an unknown test name
        ANM(scoring_method=IndependenceScore(ci_test="not_a_test")).fit(nonlinear_data)


def test_clone_preserves_scoring_method_instance():
    from sklearn.base import clone

    from pgmpy.causal_discovery.bivariate_scores import EntropyScore

    cloned = clone(ANM(scoring_method=EntropyScore(method="vasicek")))
    assert isinstance(cloned.scoring_method, EntropyScore)
    assert cloned.scoring_method.method == "vasicek"


def test_incompatible_score_instance_raises(nonlinear_data):
    from pgmpy.causal_discovery.bivariate_scores import SlopeScore

    with pytest.raises(ValueError, match="SlopeScore does not support ANM"):
        ANM(scoring_method=SlopeScore()).fit(nonlinear_data)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (pd.DataFrame({"X": [1.0, 1.0, 1.0], "Y": [1.0, 2.0, 3.0]}), "constant"),
        (pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 3.0], "Z": [2.0, 1.0, 0.0]}), "exactly two variables"),
        (pd.DataFrame({"X": [0.0, 1.0, np.nan], "Y": [1.0, 2.0, 3.0]}), None),
        (pd.DataFrame({"X": list("aabbab"), "Y": list("xyxyxy")}), "continuous"),
    ],
)
def test_invalid_input_raises(data, match):
    with pytest.raises(ValueError, match=match):
        ANM().fit(data)
