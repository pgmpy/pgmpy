from sklearn.dummy import DummyRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.models.DoubleMLRegressor import DoubleMLRegressor


def make_estimator():
    class DummyDAG:
        exposure = "x0"

        def get_adjustment_set(self):
            return ["x1", "x2"]

    return DoubleMLRegressor(
        dag=DummyDAG(),
        estimator_g=DummyRegressor(),
        estimator_m=None,
        adjustment_set=None,
        treatment_col=None,
        n_folds=5,
        random_state=0,
    )


@parametrize_with_checks([make_estimator()])
def test_sklearn_compat(estimator, check):
    check(estimator)
