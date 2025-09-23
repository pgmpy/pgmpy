import unittest

from sklearn.utils.estimator_checks import check_estimator

from pgmpy.causal_discovery import PC


class TestPCEstimatorClass(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_sklearn_compatibiltity(self):
        pc_estimator = PC()

        try:
            check_estimator(pc_estimator)
        except Exception as e:
            self.fail(f"PC estimator is not an sklearn-type Estimator: {e}")
