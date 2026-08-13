import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery import IGCI


@pytest.fixture
def nonlinear_data():
    """Generate near-deterministic data with X -> Y."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 500)
    y = x**3 + rng.normal(0, 1e-3, 500)
    return pd.DataFrame({"X": x, "Y": y})


def test_init():
    est = IGCI()
    assert est.get_params() == {
        "ref_measure": "uniform",
        "scoring_method": "slope",
    }


def test_slope_score(nonlinear_data):
    from pgmpy.causal_discovery.bivariate_scores import SlopeScore, get_bivariate_score

    score = get_bivariate_score("slope", algorithm="igci")
    assert isinstance(score, SlopeScore)
    assert score.get_tag("name") == "slope"
    assert score.get_tag("supported_algorithms") == ["igci"]

    est = IGCI(scoring_method="slope").fit(nonlinear_data)
    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ == pytest.approx(0.19854159536238275, rel=1e-4)
    assert est.backward_score_ == pytest.approx(1.3206465666212361, rel=1e-4)
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.score(true_graph=est.causal_graph_, metric="SHD") == 0

    rng = np.random.default_rng(1)
    x = np.round(rng.uniform(0, 1, 500), 2)
    y = x**3 + rng.normal(0, 1e-3, 500)
    est = IGCI(scoring_method="slope").fit(pd.DataFrame({"X": x, "Y": y}))
    assert list(est.causal_graph_.edges()) == [("X", "Y")]

    with pytest.raises(ValueError, match="two distinct x values"):
        score([1, 1], [1, 2])
    with pytest.raises(ValueError, match="non-zero x and y spacings"):
        score([1, 2], [1, 1])


def test_entropy_score(nonlinear_data):
    from pgmpy.causal_discovery.bivariate_scores import EntropyDifferenceScore, get_bivariate_score

    score = get_bivariate_score("entropy", algorithm="igci")
    assert isinstance(score, EntropyDifferenceScore)
    assert score.method == "spacing"
    assert score.get_tag("name") == "entropy"
    assert score.get_tag("supported_algorithms") == ["igci"]

    est = IGCI(scoring_method="entropy").fit(nonlinear_data)
    assert list(est.causal_graph_.edges()) == [("X", "Y")]

    x = np.linspace(0.1, 1, 100)
    y = x**3
    natural_score = score(x, y)
    base_two_score = EntropyDifferenceScore(base=2)(x, y)
    assert base_two_score == pytest.approx(natural_score / np.log(2))
    assert np.isfinite(EntropyDifferenceScore(method="vasicek", window_length=5)(x, y))

    with pytest.raises(ValueError, match="distinct observations"):
        score([0, 0, 1], [0, 1, 2])

    with pytest.raises(ValueError):
        IGCI(scoring_method=EntropyDifferenceScore(method="bogus")).fit(nonlinear_data)


def test_custom_score():
    from pgmpy.causal_discovery.bivariate_scores import get_bivariate_score

    score = lambda x, y: np.mean(y) - np.mean(x)
    assert get_bivariate_score(score, algorithm="igci") is score


def test_nan_or_tied_direction_scores_raise(nonlinear_data):
    for score in (np.nan, 0.0):
        with pytest.raises(ValueError):
            IGCI(scoring_method=lambda x, y: score).fit(nonlinear_data)


@pytest.mark.parametrize("ref_measure", ["uniform", "gaussian"])
def test_reference_measures(nonlinear_data, ref_measure):
    est = IGCI(ref_measure=ref_measure).fit(nonlinear_data)
    assert list(est.causal_graph_.edges()) == [("X", "Y")]


def test_clone_preserves_scoring_method_instance():
    from sklearn.base import clone

    from pgmpy.causal_discovery.bivariate_scores import EntropyDifferenceScore

    cloned = clone(IGCI(scoring_method=EntropyDifferenceScore(method="vasicek")))
    assert isinstance(cloned.scoring_method, EntropyDifferenceScore)
    assert cloned.scoring_method.method == "vasicek"


def test_incompatible_score_instance_raises(nonlinear_data):
    from pgmpy.causal_discovery.bivariate_scores import GaussScore

    with pytest.raises(ValueError, match="GaussScore does not support IGCI"):
        IGCI(scoring_method=GaussScore()).fit(nonlinear_data)


@pytest.mark.parametrize(
    ("estimator", "data", "match"),
    [
        (IGCI(scoring_method="gauss"), pd.DataFrame({"X": [0, 1], "Y": [0, 1]}), "IGCI"),
        (IGCI(ref_measure="bogus"), pd.DataFrame({"X": [0, 1], "Y": [0, 1]}), "ref_measure"),
        (IGCI(), pd.DataFrame({"X": [1, 1, 1], "Y": [1, 2, 3]}), "constant"),
        (IGCI(), pd.DataFrame({"X": [0, 1], "Y": [1, 2], "Z": [2, 1]}), "exactly two"),
        (IGCI(), pd.DataFrame({"X": [0, 1, np.nan], "Y": [1, 2, 3]}), None),
        (IGCI(), pd.DataFrame({"X": list("aabbab"), "Y": list("xyxyxy")}), "continuous"),
    ],
)
def test_invalid_input_raises(estimator, data, match):
    with pytest.raises(ValueError, match=match):
        estimator.fit(data)
