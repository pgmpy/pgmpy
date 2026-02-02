import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from pgmpy.metrics import CorrelationScore
from pgmpy.utils import get_example_model


@pytest.fixture
def model_and_data():
    alarm_model = get_example_model("alarm")
    alarm_data = alarm_model.simulate(int(1e4), show_progress=False)

    return alarm_model, alarm_data


def test_discrete_network(model_and_data):
    alarm_model, alarm_data = model_and_data

    for test in {
        None,
        "chi_square",
        "g_sq",
        "log_likelihood",
        "modified_log_likelihood",
    }:
        for score in {f1_score, accuracy_score}:
            corr_scorer = CorrelationScore(
                ci_test=test, score=score, return_summary=False
            )

            metric = corr_scorer(X=alarm_data, causal_graph=alarm_model)
            assert isinstance(metric, float)

            corr_scorer = CorrelationScore(
                ci_test=test, score=score, return_summary=True
            )
            metric_summary = corr_scorer(X=alarm_data, causal_graph=alarm_model)
            assert isinstance(metric_summary, pd.DataFrame)


def test_input(model_and_data):
    alarm_model, alarm_data = model_and_data

    with pytest.raises(ValueError):
        corr_scorer = CorrelationScore(ci_test="some_random_test", score=f1_score)
        corr_scorer(X=alarm_data, causal_graph=alarm_model)

    with pytest.raises(ValueError):
        corr_scorer = CorrelationScore(ci_test="chi_square", score="not_a_score")
        corr_scorer(X=alarm_data, causal_graph=alarm_model)

    with pytest.raises(ValueError):
        corr_scorer = CorrelationScore()
        corr_scorer(X=alarm_data, causal_graph="not_a_model")

    with pytest.raises(ValueError):
        alarm_data_copy = alarm_data.copy()
        alarm_data_copy.columns = range(len(alarm_data_copy.columns))
        corr_scorer = CorrelationScore(ci_test="chi_square", score=f1_score)
        corr_scorer(X=alarm_data_copy, causal_graph=alarm_model)
