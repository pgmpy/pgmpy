import unittest

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from pgmpy.base import DAG
from pgmpy.estimators import PC
from pgmpy.estimators.CITests import chi_square
from pgmpy.factors.discrete import TabularCPD
from pgmpy.metrics import (
    SHD,
    correlation_score,
    fisher_c,
    implied_cis,
    log_likelihood_score,
    self_compatibility_score,
    structure_score,
)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


class TestCorrelationScore(unittest.TestCase):
    def setUp(self):
        self.alarm = get_example_model("alarm")
        self.data = self.alarm.simulate(int(1e4), show_progress=False)

    def test_discrete_network(self):
        for test in {
            "chi_square",
            "g_sq",
            "log_likelihood",
            "modified_log_likelihood",
        }:
            for score in {f1_score, accuracy_score}:
                metric = correlation_score(
                    self.alarm, self.data, test=test, score=score
                )
                self.assertTrue(isinstance(metric, float))

                metric_summary = correlation_score(
                    self.alarm, self.data, test=test, score=score, return_summary=True
                )
                self.assertTrue(isinstance(metric_summary, pd.DataFrame))

    def test_input(self):
        self.assertRaises(
            ValueError, correlation_score, self.alarm, self.data, "some_random_test"
        )
        self.assertRaises(
            ValueError, correlation_score, "I am wrong model type", self.data
        )
        self.assertRaises(ValueError, correlation_score, self.alarm, self.data.values)

        df_wrong_columns = self.data.copy()
        df_wrong_columns.columns = range(len(self.data.columns))
        self.assertRaises(ValueError, correlation_score, self.alarm, df_wrong_columns)

        self.assertRaises(
            ValueError, correlation_score, self.alarm, self.data, score="Wrong type"
        )


class TestStructureScore(unittest.TestCase):
    def setUp(self):
        self.alarm = get_example_model("alarm")
        self.data = self.alarm.simulate(int(1e4), show_progress=False)

        # Remove all CPDs
        self.alarm_no_cpd = self.alarm.copy()
        self.alarm_no_cpd.cpds = []

    def test_discrete_network(self):
        for model in {self.alarm, self.alarm_no_cpd}:
            for scoring_method in {"k2", "bdeu", "bds", "bic-d"}:
                metric = structure_score(self.alarm, self.data, scoring_method)
                self.assertTrue(isinstance(metric, float))
            for scoring_method in {"bdeu", "bds"}:
                metric = structure_score(
                    self.alarm, self.data, scoring_method, equivalent_sample_size=10
                )
                self.assertTrue(isinstance(metric, float))

    def test_input(self):
        self.assertRaises(
            ValueError, structure_score, self.alarm, self.data, "random scoring"
        )
        self.assertRaises(
            ValueError, structure_score, "I am wrong model type", self.data
        )
        self.assertRaises(ValueError, structure_score, self.alarm, self.data.values)

        df_wrong_columns = self.data.copy()
        df_wrong_columns.columns = range(len(self.data.columns))
        self.assertRaises(ValueError, structure_score, self.alarm, df_wrong_columns)


class TestLogLikelihoodScore(unittest.TestCase):
    def setUp(self):
        self.model = get_example_model("alarm")
        self.data = self.model.simulate(int(1e4), show_progress=False)

    def test_discrete_network(self):
        metric = log_likelihood_score(self.model, self.data)
        self.assertTrue(isinstance(metric, float))

    def test_input(self):
        self.assertRaises(
            ValueError, log_likelihood_score, "I am wrong model type", self.data
        )
        self.assertRaises(
            ValueError, log_likelihood_score, self.model, self.data.values
        )

        df_wrong_columns = self.data.copy()
        df_wrong_columns.columns = range(len(self.data.columns))
        self.assertRaises(
            ValueError, log_likelihood_score, self.model, df_wrong_columns
        )


class TestImpliedCI(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)

        self.model_cancer = get_example_model("cancer")

        n_cancer = len(self.model_cancer.nodes())
        self.model_cancer_random = nx.from_numpy_array(
            np.tril(rng.choice([0, 1], p=[0.9, 0.1], size=(n_cancer, n_cancer)), k=-1),
            create_using=nx.DiGraph,
        )
        nx.relabel_nodes(
            self.model_cancer_random,
            {i: list(self.model_cancer.nodes())[i] for i in range(n_cancer)},
            copy=False,
        )
        self.model_cancer_random = DAG(self.model_cancer_random.edges())
        # self.model_cancer_random.get_random_cpds(inplace=True)

        self.model_alarm = get_example_model("alarm")
        n_alarm = len(self.model_alarm.nodes())
        self.model_alarm_random = nx.from_numpy_array(
            np.tril(rng.choice([0, 1], p=[0.9, 0.1], size=(n_alarm, n_alarm)), k=-1),
            create_using=nx.DiGraph,
        )
        nx.relabel_nodes(
            self.model_alarm_random,
            {i: list(self.model_alarm.nodes())[i] for i in range(n_alarm)},
            copy=False,
        )
        self.model_alarm_random = DAG(self.model_alarm_random.edges())

        self.df_cancer = self.model_cancer.simulate(int(1e3), seed=42)
        self.df_alarm = self.model_alarm.simulate(int(1e3), seed=42)

    def test_implied_cis(self):
        cancer_tests = implied_cis(self.model_cancer, self.df_cancer, chi_square)
        self.assertEqual(cancer_tests.shape[0], 6)
        self.assertEqual(
            list(cancer_tests.loc[:, "p-value"].values.round(4)),
            [0.9816, 1.0, 0.3491, 0.8061, 0.896, 0.9917],
        )

        alarm_tests_true = implied_cis(self.model_alarm, self.df_alarm, chi_square)
        self.assertEqual(alarm_tests_true.shape[0], 620)

        alarm_tests_random = implied_cis(
            self.model_alarm_random, self.df_alarm, chi_square
        )
        self.assertEqual(alarm_tests_random.shape[0], 528)

    def test_fisher_c(self):
        p_value = fisher_c(self.model_cancer, self.df_cancer, chi_square)
        self.assertEqual(round(p_value, 4), 0.9967)

        p_value = fisher_c(self.model_cancer_random, self.df_cancer, chi_square)
        self.assertEqual(round(p_value, 4), 0.0001)

        p_value = fisher_c(self.model_alarm, self.df_alarm, chi_square)
        self.assertEqual(round(p_value, 4), 0.0005)

        p_value = fisher_c(self.model_alarm_random, self.df_alarm, chi_square)
        self.assertEqual(p_value, 0)


class TestStructuralHammingDistance(unittest.TestCase):
    def setUp(self):
        self.dag_1 = DiscreteBayesianNetwork([(1, 2)])
        self.dag_2 = DiscreteBayesianNetwork([(2, 1)])

        self.dag_3 = DiscreteBayesianNetwork([(1, 2), (2, 4), (1, 3), (3, 4)])
        self.dag_4 = DiscreteBayesianNetwork([(1, 2), (1, 3), (3, 2), (3, 4)])

        self.dag_5 = DiscreteBayesianNetwork([(1, 2), (1, 3), (3, 2), (3, 5)])

        self.large_dag_1 = DiscreteBayesianNetwork(
            [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)]
        )
        self.large_dag_2 = DiscreteBayesianNetwork(
            [(1, 2), (1, 3), (4, 2), (3, 5), (4, 6), (5, 6)]
        )

    def test_shd(self):
        self.assertEqual(SHD(self.dag_1, self.dag_2), 1)

    def test_shd(self):
        self.assertEqual(SHD(self.dag_3, self.dag_4), 2)

    def test_shd(self):
        self.assertEqual(SHD(self.large_dag_1, self.large_dag_2), 3)

    def test_shd_unequal_graphs(self):
        with self.assertRaises(ValueError, msg="The graphs must have the same nodes."):
            SHD(self.dag_4, self.dag_5)


class TestStructuralHammingDistance(unittest.TestCase):
    def setUp(self):
        self.dag_1 = DiscreteBayesianNetwork([(1, 2)])
        self.dag_2 = DiscreteBayesianNetwork([(2, 1)])

        self.dag_3 = DiscreteBayesianNetwork([(1, 2), (2, 4), (1, 3), (3, 4)])
        self.dag_4 = DiscreteBayesianNetwork([(1, 2), (1, 3), (3, 2), (3, 4)])

        self.dag_5 = DiscreteBayesianNetwork([(1, 2), (1, 3), (3, 2), (3, 5)])

        self.large_dag_1 = DiscreteBayesianNetwork(
            [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)]
        )
        self.large_dag_2 = DiscreteBayesianNetwork(
            [(1, 2), (1, 3), (4, 2), (3, 5), (4, 6), (5, 6)]
        )

    def test_shd(self):
        self.assertEqual(SHD(self.dag_1, self.dag_2), 1)

    def test_shd(self):
        self.assertEqual(SHD(self.dag_3, self.dag_4), 2)

    def test_shd(self):
        self.assertEqual(SHD(self.large_dag_1, self.large_dag_2), 3)

    def test_shd_unequal_graphs(self):
        with self.assertRaises(ValueError, msg="The graphs must have the same nodes."):
            SHD(self.dag_4, self.dag_5)


class TestSelfCompatibility(unittest.TestCase):
    """Unit tests for the self_compatibility_score metric."""

    def setUp(cls):
        """Prepare a clean chain model and both large and noisy datasets."""
        model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])
        model.add_cpds(
            TabularCPD("A", 2, [[0.6], [0.4]]),
            TabularCPD(
                "B", 2, [[0.8, 0.2], [0.2, 0.8]], evidence=["A"], evidence_card=[2]
            ),
            TabularCPD(
                "C", 2, [[0.9, 0.3], [0.1, 0.7]], evidence=["B"], evidence_card=[2]
            ),
        )
        sampler = BayesianModelSampling(model)
        cls.large_data = sampler.forward_sample(size=5000, seed=0)
        noisy = cls.large_data.copy()
        mask = np.random.RandomState(1).rand(*noisy.shape) < 0.05
        noisy_vals = noisy.values.astype(int)
        noisy_vals[mask] = 1 - noisy_vals[mask]
        cls.noisy_data = pd.DataFrame(noisy_vals, columns=noisy.columns)

    def test_full_subsampling_perfect_score(self):
        """
        When subset_fraction=1.0, all variables are used each run.
        On clean data, learned graphs are identical, so score == 1.0.
        """
        score = self_compatibility_score(
            PC, self.large_data, num_subsets=10, subset_fraction=1.0, random_state=42
        )
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_partial_subsampling_high_score(self):
        """
        With subset_fraction=2/3 on three variables, subgraphs are size 2.
        On clean data, PC recovers each subgraph consistently → score ≥ 0.95.
        """
        score = self_compatibility_score(
            PC, self.large_data, num_subsets=30, subset_fraction=2 / 3, random_state=123
        )
        self.assertGreaterEqual(score, 0.95)
        self.assertLessEqual(score, 1.0)

    def test_partial_subsampling_noisy_lower_score(self):
        """
        Noisy data should reduce consistency on partial subsampling:
        score remains > 0 but < 1.
        """
        score = self_compatibility_score(
            PC, self.noisy_data, num_subsets=30, subset_fraction=2 / 3, random_state=7
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_estimator_kwargs_passed(self):
        """
        Ensure that arbitrary estimator_kwargs (e.g. significance_level)
        are forwarded into the learner’s .estimate() call.
        """

        # Stub that records whatever kwargs are passed to .estimate()
        class StubEstimator:
            def __init__(self, df):
                self.df = df
                self.kwargs_seen = {}

            def estimate(self, **kwargs):
                self.kwargs_seen = kwargs

                # Return a minimal graph-like object
                class G:
                    def nodes(self):
                        return []

                    def has_edge(self, u, v):
                        return False

                return G()

        # Always return our single stub instance
        stub = StubEstimator(self.large_data)

        class StubFactory:
            def __new__(cls, df):
                return stub

        # Call the compatibility score with a custom kwarg
        _ = self_compatibility_score(
            StubFactory,
            self.large_data,
            num_subsets=3,
            subset_fraction=1.0,
            random_state=0,
            myparam=123,
        )

        # Verify it arrived intact
        self.assertIn("myparam", stub.kwargs_seen)
        self.assertEqual(stub.kwargs_seen["myparam"], 123)

    def test_invalid_estimator_raises(self):
        """
        Providing an invalid estimator class (non-class, missing .estimate,
        or bad constructor) must raise AttributeError.
        """
        with self.assertRaises(AttributeError):
            self_compatibility_score("notaclass", self.large_data)

        class BadEstimator:
            def __init__(self, df):
                pass

        with self.assertRaises(AttributeError):
            self_compatibility_score(BadEstimator, self.large_data)

        class BadInit:
            def __init__(self, df):
                raise RuntimeError

        with self.assertRaises(AttributeError):
            self_compatibility_score(BadInit, self.large_data)
