import unittest

from pgmpy.base import DAG
from pgmpy.causal_discovery import PC
from pgmpy.estimators.causal_discovery import check_causal_discovery


class DummyEstimator1:
    # Missing score
    def fit(self, X):
        pass


class DummyEstimator2:
    # Missing fit
    def score(self):
        pass


class DummyEstimator3:
    # Missing causal_graph_
    def fit(self, X):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns
        self.adjacency_matrix_ = None
        return self

    def score(self):
        pass


class DummyEstimator4:
    # Missing n_features_in_
    def fit(self, X):
        self.causal_graph_ = DAG()
        self.feature_names_in_ = X.columns
        self.adjacency_matrix_ = None
        return self

    def score(self):
        pass


class DummyEstimator5:
    # Everything correct
    def fit(self, X, independencies=None):
        self.causal_graph_ = DAG()
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns
        self.adjacency_matrix_ = None
        return self

    def score(self):
        pass


class DummyEstimator6:
    # Missing feature_names_in_
    def fit(self, X):
        self.causal_graph_ = DAG()
        self.n_features_in_ = X.shape[1]
        self.adjacency_matrix_ = None
        return self

    def score(self):
        pass


class DummyEstimator7:
    # Missing adjacency_matrix_
    def fit(self, X):
        self.causal_graph_ = DAG()
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns
        return self

    def score(self):
        pass


class DummyEstimator8:
    # Fails during fit
    def fit(self, X):
        raise ValueError("Intentional error")

    def score(self):
        pass


class TestEstimatorChecks(unittest.TestCase):
    def test_missing_methods(self):
        est1 = DummyEstimator1()
        with self.assertRaises(AttributeError) as context:
            check_causal_discovery(est1)
        self.assertTrue("must have a 'score' method" in str(context.exception))
        est1.fit(None)

        est2 = DummyEstimator2()
        with self.assertRaises(AttributeError) as context:
            check_causal_discovery(est2)
        self.assertTrue("must have a 'fit' method" in str(context.exception))
        est2.score()

    def test_missing_attributes(self):
        est3 = DummyEstimator3()
        with self.assertRaises(AttributeError) as context:
            check_causal_discovery(est3)
        self.assertTrue(
            "must set the internal attribute 'causal_graph_'" in str(context.exception)
        )
        est3.score()

        est4 = DummyEstimator4()
        with self.assertRaises(AttributeError) as context:
            check_causal_discovery(est4)
        self.assertTrue(
            "must set the internal attribute 'n_features_in_'" in str(context.exception)
        )
        est4.score()

        est6 = DummyEstimator6()
        with self.assertRaises(AttributeError) as context:
            check_causal_discovery(est6)
        self.assertTrue(
            "must set the internal attribute 'feature_names_in_'"
            in str(context.exception)
        )
        est6.score()

        est7 = DummyEstimator7()
        with self.assertRaises(AttributeError) as context:
            check_causal_discovery(est7)
        self.assertTrue(
            "must set the internal attribute 'adjacency_matrix_'"
            in str(context.exception)
        )
        est7.score()

    def test_fit_exception(self):
        est8 = DummyEstimator8()
        with self.assertRaises(RuntimeError) as context:
            check_causal_discovery(est8)
        self.assertTrue(
            "failed when run on a dummy dataset with error" in str(context.exception)
        )
        est8.score()

    def test_valid_estimator(self):
        est5 = DummyEstimator5()
        self.assertTrue(check_causal_discovery(est5))
        est5.score()

    def test_real_estimator(self):
        pc = PC()
        # The test checks whether PC correctly adheres to the interface without throwing errors
        self.assertTrue(check_causal_discovery(pc))
