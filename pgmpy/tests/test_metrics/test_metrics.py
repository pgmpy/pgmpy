import unittest

import networkx as nx
import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.metrics import accuracy_score, f1_score

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators.CITests import chi_square
from pgmpy.metrics import (
    SHD,
    correlation_score,
    fisher_c,
    implied_cis,
    log_likelihood_score,
    structure_score,
)
from pgmpy.models import DiscreteBayesianNetwork
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
        for model in [self.alarm, self.alarm_no_cpd]:
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


@unittest.skipUnless(
    _check_soft_dependencies("torch", severity="none"),
    reason="execute only if required dependency present",
)
class TestLogLikelihoodScoreTorch(TestLogLikelihoodScore):
    def setUp(self):
        self.original_backend = config.get_backend()
        config.set_backend("torch")
        super().setUp()

    def tearDown(self):
        config.set_backend(self.original_backend)


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

    def test_fisher_c_rmsea(self):
        (p_value, rmsea) = fisher_c(
            self.model_cancer, self.df_cancer, chi_square, compute_rmsea=True
        )
        self.assertEqual(round(p_value, 4), 0.9967)
        self.assertEqual(round(rmsea, 4), 0)

        (p_value, rmsea) = fisher_c(
            self.model_cancer_random, self.df_cancer, chi_square, compute_rmsea=True
        )
        self.assertEqual(round(p_value, 4), 0.0001)
        self.assertEqual(round(rmsea, 4), 0.0602)

        (p_value, rmsea) = fisher_c(
            self.model_alarm, self.df_alarm, chi_square, compute_rmsea=True
        )
        self.assertEqual(round(p_value, 4), 0.0005)
        self.assertEqual(round(rmsea, 4), 0.0117)

        (p_value, rmsea) = fisher_c(
            self.model_alarm_random, self.df_alarm, chi_square, compute_rmsea=True
        )
        self.assertEqual(round(p_value, 4), 0)
        self.assertEqual(round(rmsea, 4), 0.0476)


class TestStructuralHammingDistance(unittest.TestCase):
    def test_shd1(self):
        dag1 = DiscreteBayesianNetwork([(1, 2)])
        dag2 = DiscreteBayesianNetwork([(2, 1)])
        self.assertEqual(SHD(dag1, dag2), 1)

    def test_shd2(self):
        dag1 = DiscreteBayesianNetwork([(1, 2), (2, 4), (1, 3), (3, 4)])
        dag2 = DiscreteBayesianNetwork([(1, 2), (1, 3), (3, 2), (3, 4)])
        self.assertEqual(SHD(dag1, dag2), 2)

    def test_shd3(self):
        dag1 = DiscreteBayesianNetwork([(1, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)])
        dag2 = DiscreteBayesianNetwork([(1, 2), (1, 3), (4, 2), (3, 5), (4, 6), (5, 6)])
        self.assertEqual(SHD(dag1, dag2), 3)

    def test_shd_isolated_nodes(self):
        dag1 = DiscreteBayesianNetwork([(1, 2)])
        dag1.add_nodes_from([3])
        dag2 = DiscreteBayesianNetwork([(1, 2), (2, 3)])

        self.assertEqual(SHD(dag1, dag2), 1)
        self.assertEqual(SHD(dag2, dag1), 1)

    def test_shd_mixed_differences(self):
        dag1 = DiscreteBayesianNetwork([(1, 2), (2, 3), (2, 4), (4, 5), (6, 5), (7, 8)])
        dag1.add_nodes_from([9, 10])
        dag2 = DiscreteBayesianNetwork(
            [(1, 2), (2, 4), (5, 4), (6, 5), (8, 7), (9, 10)]
        )
        dag2.add_nodes_from([3, 7])

        self.assertEqual(SHD(dag1, dag2), 4)
        self.assertEqual(SHD(dag2, dag1), 4)

    def test_shd_unequal_graphs(self):
        dag1 = DiscreteBayesianNetwork([(1, 2), (1, 3), (3, 2), (3, 4)])
        dag2 = DiscreteBayesianNetwork([(1, 2), (1, 3), (3, 2), (3, 5)])

        with self.assertRaises(ValueError, msg="The graphs must have the same nodes."):
            SHD(dag1, dag2)
