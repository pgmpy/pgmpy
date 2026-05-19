from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery.TOPIC import TOPIC


def _score_obj(score_fn):
    """Wrap a callable ``score_fn(node, parents)`` in an object exposing ``local_score``."""
    return SimpleNamespace(local_score=score_fn)


@pytest.fixture
def fake_data():
    np.random.seed(42)
    return pd.DataFrame(
        np.random.random((1000, 4)),
        columns=["A", "B", "C", "D"],
    )


""" 1. Compatibility Tests """


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [TOPIC(return_type="dag")],
    expected_failed_checks=expected_failed_checks,
)
def test_topic_compatibility(estimator, check):
    check(estimator)


""" 2. Unit Tests (fake score function) """


def test_find_removable_edge_single_parent():
    topic = TOPIC()

    result = topic._find_removable_edge(parents=["A"], child="X", score=_score_obj(lambda node, parents: 0.0))

    assert result is None


def test_find_removable_edge_best_parent():
    topic = TOPIC()

    def score_fn(child, parents):
        s = set(parents)
        if s == {"A", "B", "C"}:
            return 30.0
        if s == {"B", "C"}:
            return 10.0
        if s == {"A", "C"}:
            return 25.0
        if s == {"A", "B"}:
            return 50.0

    # harms: A=-20, B=-5, C=+20 — C is the only non-harmful removal.
    result = topic._find_removable_edge(
        parents=["A", "B", "C"],
        child="X",
        score=_score_obj(score_fn),
    )
    assert result == "C"


def test_find_removable_edge_no_removable_candidate():
    topic = TOPIC()

    def score_fn(child, parents):
        return 100.0 - (3 - len(parents)) * 10.0

    result = topic._find_removable_edge(
        parents=["A", "B", "C"],
        child="X",
        score=_score_obj(score_fn),
    )
    assert result is None


def test_find_removable_edge_allows_small_negative_harm_due_to_float_noise():
    topic = TOPIC()

    def score_fn2(child, parents):
        s = set(parents)
        if s == {"A", "B"}:
            return 1.0
        if s == {"B"}:
            return 1.0 - 1e-12  # removing A: tiny negative harm, allowed by noise tolerance
        return 0.0  # removing B leaves only A: large negative harm, not allowed

    result = topic._find_removable_edge(parents=["A", "B"], child="X", score=_score_obj(score_fn2))
    assert result == "A"


""" 3. Smoke Test (fake data) """


def test_fit_scoring_methods(fake_data):
    est = TOPIC()
    dag = est.fit(fake_data)
    assert dag is not None
    assert est.n_features_in_ == fake_data.shape[1]
    assert len(est.feature_names_in_) == len(np.asarray(fake_data.columns, dtype=object))


@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
@pytest.mark.parametrize("show_progress", [True, False])
@pytest.mark.parametrize("return_type", ["dag", "pdag"])
def test_arguments(fake_data, scoring_method, show_progress, return_type):
    est = TOPIC(
        scoring_method=scoring_method,
        show_progress=show_progress,
        return_type=return_type,
    )
    _ = est.fit(fake_data)
