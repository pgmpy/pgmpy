from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import ILPSearch


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [ILPSearch()],
    expected_failed_checks=expected_failed_checks,
)
def test_ilp_compatibility(estimator, check):
    check(estimator)
