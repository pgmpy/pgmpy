import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.utils import get_example_model


class TestLGBNMethods(unittest.TestCase):
    def setUp(self):
        self.model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        self.cpd1 = LinearGaussianCPD(variable="x1", beta=[1], std=4)
        self.cpd2 = LinearGaussianCPD(
            variable="x2", beta=[-5, 0.5], std=4, evidence=["x1"]
        )
        self.cpd3 = LinearGaussianCPD(
            variable="x3", beta=[4, -1], std=3, evidence=["x2"]
        )

    def test_cpds_simple(self):
        self.assertEqual("x1", self.cpd1.variable)
        self.assertEqual(4, self.cpd1.std)
        self.assertEqual([1], self.cpd1.beta)

        self.model.add_cpds(self.cpd1)
        cpd = self.model.get_cpds("x1")
        self.assertEqual(cpd.variable, self.cpd1.variable)
        self.assertEqual(cpd.std, self.cpd1.std)
        self.assertEqual(cpd.beta, self.cpd1.beta)

    def test_add_cpds(self):
        self.model.add_cpds(self.cpd1)
        cpd = self.model.get_cpds("x1")
        self.assertEqual(cpd.variable, self.cpd1.variable)
        self.assertEqual(cpd.std, self.cpd1.std)

        self.model.add_cpds(self.cpd2)
        cpd = self.model.get_cpds("x2")
        self.assertEqual(cpd.variable, self.cpd2.variable)
        self.assertEqual(cpd.std, self.cpd2.std)
        self.assertEqual(cpd.evidence, self.cpd2.evidence)

        self.model.add_cpds(self.cpd3)
        cpd = self.model.get_cpds("x3")
        self.assertEqual(cpd.variable, self.cpd3.variable)
        self.assertEqual(cpd.std, self.cpd3.std)
        self.assertEqual(cpd.evidence, self.cpd3.evidence)

        tab_cpd = TabularCPD(
            "grade",
            3,
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ],
            evidence=["diff", "intel"],
            evidence_card=[2, 3],
        )
        self.assertRaises(ValueError, self.model.add_cpds, tab_cpd)
        self.assertRaises(ValueError, self.model.add_cpds, 1)
        self.assertRaises(ValueError, self.model.add_cpds, 1, tab_cpd)

    def test_to_joint_gaussian(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        mean, cov = self.model.to_joint_gaussian()
        np_test.assert_array_almost_equal(mean, np.array([1.0, -4.5, 8.5]), decimal=3)
        np_test.assert_array_almost_equal(
            cov,
            np.array([[4.0, 2.0, -2.0], [2.0, 5.0, -5.0], [-2.0, -5.0, 8.0]]),
            decimal=3,
        )

    def test_check_model(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        self.assertEqual(self.model.check_model(), True)

        self.model.add_edge("x1", "x4")
        cpd4 = LinearGaussianCPD("x4", [4, -1], 3, ["x2"])
        self.model.add_cpds(cpd4)

        self.assertRaises(ValueError, self.model.check_model)

    def test_not_implemented_methods(self):
        self.assertRaises(ValueError, self.model.get_cardinality, "x1")
        self.assertRaises(NotImplementedError, self.model.to_markov_model)
        self.assertRaises(
            NotImplementedError, self.model.is_imap, [[1, 2, 3], [1, 5, 6]]
        )

    def test_simulate(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        df_cont = self.model.simulate(n=10000, seed=42)

        # Same model in terms of equations
        rng = np.random.default_rng(seed=42)
        x1 = 1 + rng.normal(0, 2, 10000)
        x2 = -5 + 0.5 * x1 + rng.normal(0, 2, 10000)
        x3 = 4 + -1 * x2 + rng.normal(0, np.sqrt(3), 10000)
        df_equ = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        np_test.assert_array_almost_equal(df_cont.mean(), df_equ.mean(), decimal=1)
        np_test.assert_array_almost_equal(df_cont.cov(), df_equ.cov(), decimal=1)

    def test_fit(self):
        # Test fit on a simple model
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        df = self.model.simulate(int(1e5), seed=42)
        new_model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        new_model.fit(df, method="mle")

        for node in self.model.nodes():
            cpd_orig = self.model.get_cpds(node)
            cpd_est = new_model.get_cpds(node)

            self.assertEqual(cpd_orig.variable, cpd_est.variable)
            self.assertEqual(round(cpd_orig.std, 1), round(cpd_est.std, 1))
            self.assertEqual(round(cpd_orig.beta[0], 1), round(cpd_est.beta[0], 1))

            for index, evid_var in enumerate(cpd_orig.evidence):
                est_index = cpd_est.evidence.index(evid_var)
                self.assertEqual(
                    round(cpd_orig.beta[index + 1], 1),
                    round(cpd_est.beta[est_index + 1], 1),
                )

        # Test fit on the alarm model
        model = get_example_model("alarm")
        model_lin = LinearGaussianBayesianNetwork(model.edges())
        cpds = model_lin.get_random_cpds()
        model_lin.add_cpds(*cpds)
        df = model_lin.simulate(int(1e6), seed=42)

        new_model_lin = LinearGaussianBayesianNetwork(model.edges())
        new_model_lin.fit(df, method="mle")

        for node in model_lin.nodes():
            cpd_orig = model_lin.get_cpds(node)
            cpd_est = new_model_lin.get_cpds(node)

            self.assertEqual(cpd_orig.variable, cpd_est.variable)
            self.assertTrue(abs(cpd_orig.std - cpd_est.std) < 0.1)
            self.assertTrue(abs(cpd_orig.beta[0] - cpd_est.beta[0]) < 0.1)

            for index, evid_var in enumerate(cpd_orig.evidence):
                est_index = cpd_est.evidence.index(evid_var)
                self.assertTrue(
                    abs(cpd_orig.beta[index + 1] - cpd_est.beta[est_index + 1]) < 0.1
                )

    def test_predict(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        df = self.model.simulate(int(10), seed=42)
        df = df.drop("x2", axis=1)
        variables, mu, cov = self.model.predict(df)
        self.assertEqual(variables, ["x2"])
        self.assertEqual(mu.shape, (10, 1))
        self.assertTrue(
            np.allclose(
                mu.round(2).squeeze(),
                [-5.31, -5.63, -4.71, -3.3, -4.82, -2.61, -5.98, -3.25, -3.94, -5.32],
            )
        )
        self.assertEqual(cov.round(2).squeeze(), 1.71)

        # Test predict on the alarm model
        model = get_example_model("alarm")
        model_lin = LinearGaussianBayesianNetwork(model.edges())
        cpds = model_lin.get_random_cpds(seed=42)
        model_lin.add_cpds(*cpds)
        df = model_lin.simulate(int(5), seed=42)

        variables, mu, cov = model_lin.predict(df.drop(["HISTORY", "CO"], axis=1))
        self.assertEqual(mu.shape, (5, 2))
        expected_mu = np.array(
            [[0.89, 0.86], [0.04, 0.85], [-1.37, 0.91], [-1.23, 0.85], [-2.08, 0.85]]
        )
        expected_cov = np.array([[0.42, 0.19], [1.94, 0.18]])

        if variables == ["HISTORY", "CO"]:
            expected_mu = expected_mu[:, [1, 0]]
            expected_cov = expected_cov.T
            expected_cov[0, 0], expected_cov[1, 1] = (
                expected_cov[1, 1],
                expected_cov[0, 0],
            )
        # TODO: Check why the following are failing on Github action
        # self.assertTrue(np.allclose(mu, expected_mu, atol=1e-1))
        # self.assertEqual(cov.shape, (2, 2))
        # self.assertTrue(np.allclose(cov, expected_cov, atol=1e-1))

    def test_get_random_cpds(self):
        model = get_example_model("alarm")
        model_lin = LinearGaussianBayesianNetwork(model.edges())
        cpds = model_lin.get_random_cpds()
        self.assertEqual(len(cpds), len(model.nodes()))

    def test_get_random(self):
        model1 = LinearGaussianBayesianNetwork.get_random(n_nodes=10, edge_prob=0.8)
        model2 = LinearGaussianBayesianNetwork.get_random(n_nodes=10, edge_prob=0.1)
        self.assertNotEqual(model1.edges(), model2.edges())
        self.assertIsInstance(
            model1, LinearGaussianBayesianNetwork, "Incorrect instance"
        )
        self.assertIsInstance(
            model2, LinearGaussianBayesianNetwork, "Incorrect instance"
        )

        node_names = ["a", "aa", "aaa", "aaaa", "aaaaa"]
        model3 = LinearGaussianBayesianNetwork.get_random(
            n_nodes=5, edge_prob=0.5, node_names=node_names
        )
        self.assertEqual(len(model3.nodes()), 5)
        self.assertEqual(sorted(model3.nodes()), node_names)
        self.assertEqual(len(model3.cpds), 5)
        self.assertIsInstance(
            model3, LinearGaussianBayesianNetwork, "Incorrect instance"
        )

    def tearDown(self):
        del self.model, self.cpd1, self.cpd2, self.cpd3


class TestLGBNCreation(unittest.TestCase):
    def test_class_init_with_adj_matrix_dict_of_dict(self):
        adj = {"a": {"b": 4, "c": 3}, "b": {"c": 2}}
        self.graph = LinearGaussianBayesianNetwork(adj, latents=set(["a"]))
        self.assertEqual(self.graph.latents, set("a"))
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])
        self.assertEqual(self.graph.adj["a"]["c"]["weight"], 3)

    def test_class_init_with_adj_matrix_dict_of_list(self):
        adj = {"a": ["b", "c"], "b": ["c"]}
        self.graph = LinearGaussianBayesianNetwork(adj, latents=set(["a"]))
        self.assertEqual(self.graph.latents, set("a"))
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])

    def test_class_init_with_pd_adj_df(self):
        df = pd.DataFrame([[0, 3], [0, 0]])
        self.graph = LinearGaussianBayesianNetwork(df, latents=set([0]))
        self.assertEqual(self.graph.latents, set([0]))
        self.assertListEqual(sorted(self.graph.nodes()), [0, 1])
        self.assertEqual(self.graph.adj[0][1]["weight"], {"weight": 3})


class TestDAGParser(unittest.TestCase):
    def test_from_lavaan(self):
        model_str = "d ~ i"
        model_from_str = LinearGaussianBayesianNetwork.from_lavaan(string=model_str)
        expected_edges = set([("i", "d")])
        self.assertEqual(set(model_from_str.edges()), expected_edges)

    def test_from_dagitty(self):
        model_str = """dag{ smoking -> "carry matches" }"""
        model_from_str = LinearGaussianBayesianNetwork.from_dagitty(string=model_str)
        expected_edges = set([("smoking", "carry matches")])
        self.assertEqual(set(model_from_str.edges()), expected_edges)
