import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery.TOPIC import TOPIC


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


""" 2. Smoke Test (fake data) """


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
