from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import PC


def make_estimator():
    return PC()


@parametrize_with_checks([make_estimator()])
def test_pc_compatibility(estimator, check):
    check(estimator)
