import unittest
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from pgmpy.base import DAG
from pgmpy.estimators.CITests import chi_square
from pgmpy.metrics import (
    SHD,
    correlation_score,
    fisher_c,
    implied_cis,
    log_likelihood_score,
    self_compatibility_graphical,
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


class TestGraphicalSelfCompatibility(unittest.TestCase):
    """
    Tests for pgmpy.metrics.self_compatibility_graphical, verifying:
      - Perfect agreement (mean SHD = 0)
      - Systematic edge flips (mean SHD = 2)
      - Proper forwarding of estimator kwargs
    """

    @classmethod
    def setUpClass(cls):
        # A simple A→B→C ground truth
        cls.full_dag = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])
        rng = np.random.RandomState(0)
        cls.data = pd.DataFrame(
            {
                "A": rng.randint(2, size=100),
                "B": rng.randint(2, size=100),
                "C": rng.randint(2, size=100),
            }
        )

    def test_perfect_compatibility(self):
        """
        If joint and all marginals always return the true DAG:
        SHD for each subset = 0 ⇒ mean = 0.0.
        """

        def perfect_factory(df):
            # always returns full_dag
            return SimpleNamespace(estimate=lambda **kw: self.full_dag)

        # Run compatibility
        score = self_compatibility_graphical(
            perfect_factory,
            self.data,
            num_subsets=10,
            subset_fraction=0.5,
            random_state=1,
        )
        # Expect zero distance everywhere
        self.assertEqual(score, 0.0)

    def test_flip_compatibility(self):
        """
        When the joint fit is correct but every marginal is flipped,
        each subset’s SHD should be 2 (two edge reversals), so the mean = 2.0.
        """
        # Capture the ground-truth edges for A→B and B→C
        full_edges = list(self.full_dag.edges())  # [('A','B'), ('B','C')]
        state = {"calls": 0}

        def flip_factory(df):
            cols = list(df.columns)  # e.g. ["A","B","C"]

            def estimate(**kw):
                # Track how many times estimate() is called
                state["calls"] += 1

                # Build a DAG on the same node set
                dag = DiscreteBayesianNetwork([])
                # Add all nodes
                for node in cols:
                    dag.add_node(node)

                # First call → joint → correct orientation
                if state["calls"] == 1:
                    for u, v in full_edges:
                        dag.add_edge(u, v)
                else:
                    # Subsequent calls → marginals → flipped orientation
                    for u, v in full_edges:
                        dag.add_edge(v, u)

                return dag

            return SimpleNamespace(estimate=estimate)

        # Compute graphical compatibility over full-node subsets
        score = self_compatibility_graphical(
            flip_factory,
            self.data,
            num_subsets=10,
            subset_fraction=1.0,  # use all nodes each time
            random_state=0,
        )

        # Now every marginal is the reverse of the joint → SHD = 2 per subset
        self.assertAlmostEqual(score, 2.0, places=6)

    def test_kwargs_forwarded(self):
        """
        Ensure that arbitrary kwargs (e.g. alpha, foo) are passed through
        exactly once to .estimate() when no marginal runs occur (num_subsets=0).
        """
        seen = {}

        def record_factory(df):
            def estimate(**kw):
                seen.update(kw)
                # We can return anything since no subsets will be processed
                return DiscreteBayesianNetwork([])

            return SimpleNamespace(estimate=estimate)

        # Use num_subsets=0 so we only invoke the joint estimator once
        _ = self_compatibility_graphical(
            record_factory,
            self.data,
            num_subsets=0,
            subset_fraction=1.0,
            random_state=3,
            alpha=0.01,
            foo="bar",
        )
        self.assertIn("alpha", seen)
        self.assertEqual(seen["alpha"], 0.01)
        self.assertIn("foo", seen)
        self.assertEqual(seen["foo"], "bar")
