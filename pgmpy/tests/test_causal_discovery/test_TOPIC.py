import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery.TOPIC import TOPIC


def make_estimator():
    return TOPIC()


@parametrize_with_checks([make_estimator()])
def test_topic_compatibility(estimator, check):
    check(estimator)


def fake_score_fn(X, Y, Z=[], **kwargs):
    pass


@pytest.fixture
def fake_data():
    np.random.seed(42)
    return pd.DataFrame(np.random.random((1000, 4)), columns=["A", "B", "C", "D"])
