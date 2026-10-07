import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.parameterization import LinearGaussianCPD, LinearGaussianOLS, TabularMLE

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)

# y = 1 + 2x plus residuals [0.5, -0.5, -0.5, 0.5], which are orthogonal to the intercept and to x, so least squares
# gives exactly beta = [1, 2], with a residual sum of squares of 1.
PARENTS = np.array([[0.0], [1.0], [2.0], [3.0]])
TARGET = np.array([1.5, 2.5, 4.5, 7.5])


def test_tags():
    # A local estimator of a LinearGaussianCPD, which passes it the rows and their weights.
    assert LinearGaussianOLS.get_class_tag("object_type") == "local_estimator"
    assert LinearGaussianOLS.get_class_tag("name") == "linear_gaussian_ols"
    assert LinearGaussianOLS.get_class_tag("parameterization") == ["linear_gaussian_cpd"]
    assert LinearGaussianOLS.get_class_tag("supports_weighted_data") is True


class TestLinearGaussianOLS:
    def test_estimate(self):
        # The unbiased std divides the residual sum of squares by the rows minus the coefficients, 4 - 2; "mle" divides
        # by the rows.
        beta, std = LinearGaussianOLS().estimate(PARENTS, TARGET, None)
        np.testing.assert_allclose(beta, [1.0, 2.0])
        assert std == pytest.approx(np.sqrt(1 / 2))
        assert LinearGaussianOLS(std_estimator="mle").estimate(PARENTS, TARGET, None)[1] == pytest.approx(0.5)

        # A root has the mean, 4, and deviations [-2.5, -1.5, 0.5, 3.5] from it: a sum of squares of 21.
        beta, std = LinearGaussianOLS().estimate(np.empty((4, 0)), TARGET, None)
        np.testing.assert_allclose(beta, [4.0])
        assert std == pytest.approx(np.sqrt(21 / 3))

        # A weight of 2 counts a row twice.
        weighted = LinearGaussianOLS().estimate(PARENTS, TARGET, np.array([2.0, 1.0, 1.0, 1.0]))
        twice = LinearGaussianOLS().estimate(PARENTS[[0, 0, 1, 2, 3]], TARGET[[0, 0, 1, 2, 3]], None)
        np.testing.assert_allclose(weighted[0], twice[0])
        assert weighted[1] == pytest.approx(twice[1])

    def test_errors(self):
        with pytest.raises(ValueError, match="std_estimator must be"):
            LinearGaussianOLS(std_estimator="unbias")
        # The unbiased std needs more rows than coefficients.
        with pytest.raises(ValueError, match="total weight"):
            LinearGaussianOLS().estimate(PARENTS[:2], TARGET[:2], None)


class TestLinearGaussianCPDEstimator:
    def test_estimator_parameter(self):
        # Least squares with the unbiased std is the default, and the estimator must be a linear Gaussian one.
        X, y = pd.DataFrame({"x": PARENTS[:, 0]}), pd.Series(TARGET, name="y")
        assert LinearGaussianCPD().fit(X, y) == LinearGaussianCPD(estimator=LinearGaussianOLS()).fit(X, y)
        assert LinearGaussianCPD(estimator=LinearGaussianOLS(std_estimator="mle")).fit(X, y).std_ == pytest.approx(0.5)
        with pytest.raises(TypeError, match="linear Gaussian estimator"):
            LinearGaussianCPD(estimator=TabularMLE())

    def test_weights_follow_the_estimator(self):
        # A LinearGaussianCPD takes sample weights only if its estimator does.
        class UnweightedOLS(LinearGaussianOLS):
            _tags = {"name": "unweighted_ols", "supports_weighted_data": False}

        cpd = LinearGaussianCPD(estimator=UnweightedOLS())
        assert cpd.get_tag("supports_weighted_data") is False
        with pytest.raises(ValueError, match="does not support sample_weight"):
            cpd.fit(None, pd.Series(TARGET, name="y"), sample_weight=np.ones(4))
