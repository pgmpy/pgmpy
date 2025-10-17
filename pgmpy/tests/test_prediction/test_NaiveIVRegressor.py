from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.prediction.NaivelVRegressor import NaiveIVRegressor


def make_estimator():
    """Create a valid estimator for sklearn compatibility tests."""
    G = DAG(
        [(0, 1), (0, 2)], roles={"exposure": [0], "instrument": [1], "outcome": [2]}
    )
    return NaiveIVRegressor(
        causal_graph=G,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
    )


@parametrize_with_checks([make_estimator()])
def test_sklearn_compatibility(estimator, check):
    """Test sklearn compatibility using parametrize_with_checks."""
    check(estimator)
