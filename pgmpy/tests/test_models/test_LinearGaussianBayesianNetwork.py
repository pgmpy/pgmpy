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

    def test_remove_cpds(self):
        model = LinearGaussianBayesianNetwork([("X1", "X2"), ("X2", "X3")])

        cpd_X1 = LinearGaussianCPD("X1", beta=[1.0], std=2.0)
        cpd_X2 = LinearGaussianCPD("X2", beta=[-1.0, 0.5], std=1.5, evidence=["X1"])
        cpd_X3 = LinearGaussianCPD("X3", beta=[0.0, 1.0], std=1.0, evidence=["X2"])

        model.add_cpds(cpd_X1, cpd_X2, cpd_X3)

        self.assertEqual(len(model.get_cpds()), 3)

        model.remove_cpds(cpd_X2, cpd_X3)

        remaining_cpds = model.get_cpds()
        self.assertEqual(len(remaining_cpds), 1)
        self.assertEqual(remaining_cpds[0], cpd_X1)

        self.assertNotIn(cpd_X2, remaining_cpds)
        self.assertNotIn(cpd_X3, remaining_cpds)

    def test_copy(self):
        model = LinearGaussianBayesianNetwork([("A", "B"), ("B", "C")])
        cpd_a = LinearGaussianCPD(variable="A", beta=[1], std=4)
        cpd_b = LinearGaussianCPD(variable="B", beta=[-5, 0.5], std=4, evidence=["A"])
        cpd_c = LinearGaussianCPD(variable="C", beta=[4, -1], std=3, evidence=["B"])

        model.add_cpds(cpd_a, cpd_b, cpd_c)

        copy_model = model.copy()

        assert set(copy_model.nodes()) == set(model.nodes())
        assert set(copy_model.edges()) == set(model.edges())

        original_cpds = model.get_cpds()
        copied_cpds = copy_model.get_cpds()

        assert len(original_cpds) == len(copied_cpds)
        for orig_cpd, copied_cpd in zip(original_cpds, copied_cpds):
            assert orig_cpd.variable == copied_cpd.variable
            assert orig_cpd.evidence == copied_cpd.evidence
            np_test.assert_array_almost_equal(orig_cpd.beta, copied_cpd.beta)
            assert orig_cpd.std == copied_cpd.std

        model.remove_cpds(cpd_b)
        assert len(model.get_cpds()) == 2
        assert len(copy_model.get_cpds()) == 3

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
        df_cont = self.model.simulate(n_samples=10000, seed=42)

        # Same model in terms of equations
        rng = np.random.default_rng(seed=42)
        x1 = 1 + rng.normal(0, 2, 10000)
        x2 = -5 + 0.5 * x1 + rng.normal(0, 2, 10000)
        x3 = 4 + -1 * x2 + rng.normal(0, np.sqrt(3), 10000)
        df_equ = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        np_test.assert_array_almost_equal(df_cont.mean(), df_equ.mean(), decimal=1)
        np_test.assert_array_almost_equal(df_cont.cov(), df_equ.cov(), decimal=1)

    def test_simulate_with_evidence(self):

        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        evidence = {"x1": 0}
        df = self.model.simulate(n_samples=10000, seed=42, evidence=evidence)

        missing_vars, mean_cond, cov_cond = self.model.predict(pd.DataFrame([evidence]))
        sorted_indices = np.argsort(missing_vars)
        missing_vars = [missing_vars[i] for i in sorted_indices]
        mean_cond = mean_cond[:, sorted_indices]
        cov_cond = cov_cond[sorted_indices][:, sorted_indices]

        rng = np.random.default_rng(seed=42)
        samples = rng.multivariate_normal(mean=mean_cond[0], cov=cov_cond, size=10000)
        df_equ = pd.DataFrame(samples, columns=missing_vars)

        np_test.assert_array_almost_equal(
            df.mean()[["x2", "x3"]], df_equ.mean(), decimal=5
        )
        np_test.assert_array_almost_equal(
            df.cov()[["x2", "x3"]].loc[["x2", "x3"]], df_equ.cov(), decimal=5
        )

    def test_simulate_with_intervention(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        do = {"x2": 1.0}
        df = self.model.simulate(n_samples=10000, seed=42, do=do)

        rng = np.random.default_rng(seed=42)
        x1 = 1 + rng.normal(0, 2, 10000)
        x2 = np.full(10000, 1.0)
        x3 = 4 + -1 * x2 + rng.normal(0, np.sqrt(3), 10000)
        df_equ = pd.DataFrame({"x1": x1, "x3": x3, "x2": do["x2"]})

        np_test.assert_array_almost_equal(df.mean(), df_equ.mean(), decimal=1)
        np_test.assert_array_almost_equal(df.cov(), df_equ.cov(), decimal=1)

    def test_simulate_against_manual_results(self):
        model = LinearGaussianBayesianNetwork(
            [("X1", "X2"), ("X1", "X3"), ("X2", "X3")]
        )

        cpd_X1 = LinearGaussianCPD("X1", beta=[0.0], std=1.0, evidence=[])
        cpd_X2 = LinearGaussianCPD("X2", beta=[0.0, 2.0], std=1.0, evidence=["X1"])
        cpd_X3 = LinearGaussianCPD(
            "X3", beta=[0.0, -1.0, 0.5], std=1.0, evidence=["X1", "X2"]
        )

        model.add_cpds(cpd_X1, cpd_X2, cpd_X3)

        df = model.simulate(n_samples=100000, seed=42, do={"X1": 1.0})

        E_X1 = 1.0
        E_X2 = 2.0 * E_X1
        E_X3 = -E_X1 + 0.5 * E_X2

        expected_mean = np.array([E_X1, E_X2, E_X3])

        expected_cov = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.25]])

        sim_mean = df[["X1", "X2", "X3"]].mean().values
        sim_cov = df[["X1", "X2", "X3"]].cov().values

        np_test.assert_array_almost_equal(sim_mean, expected_mean, decimal=1)
        np_test.assert_array_almost_equal(sim_cov, expected_cov, decimal=1)

    def test_simulate_raises_for_invalid_do_and_evidence_nodes(self):
        model = LinearGaussianBayesianNetwork([("A", "B")])
        cpd_a = LinearGaussianCPD("A", beta=[1], std=1.0)
        cpd_b = LinearGaussianCPD("B", beta=[0.5, 2.0], std=1.0, evidence=["A"])
        model.add_cpds(cpd_a, cpd_b)

        with self.assertRaisesRegex(ValueError, "do-nodes.*not present.*X"):
            model.simulate(n_samples=100, do={"X": 5.0})

        with self.assertRaisesRegex(ValueError, "evidence-nodes.*not present.*Y"):
            model.simulate(n_samples=100, evidence={"Y": 1.0})

        with self.assertRaisesRegex(ValueError, "can't be in both do and evidence.*A"):
            model.simulate(n_samples=100, do={"A": 1.0}, evidence={"A": 2.0})

    def test_fit(self):
        # Test fit on a simple model
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        df = self.model.simulate(n_samples=int(1e5), seed=42)
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
        df = model_lin.simulate(n_samples=int(1e6), seed=42)

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
        df = self.model.simulate(n_samples=int(10), seed=42)
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
        df = model_lin.simulate(n_samples=int(5), seed=42)

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

    def test_from_dagitty_DAG_ctor(self):
        from pgmpy.base import DAG

        # Adapted from https://www.dagitty.net/manual-3.x.pdf#page=4 section 3.1 with beta modified
        model_str = """dag{
        "carry matches" [latent]
        cancer [outcome]
        smoking -> "carry matches" [beta=0.2]
        smoking -> cancer [beta=0.5]
        "carry matches" -> cancer
        }"""
        model_from_str = DAG.from_dagitty(model_str)
        self.assertIsInstance(model_from_str, LinearGaussianBayesianNetwork)
        self.assertEqual(
            sorted(model_from_str.nodes()), ["cancer", "carry matches", "smoking"]
        )
        expected_edges = set(
            [
                ("smoking", "carry matches"),
                ("smoking", "cancer"),
                ("carry matches", "cancer"),
            ]
        )
        self.assertEqual(set(model_from_str.edges()), expected_edges)
        self.assertEqual(model_from_str.check_model(), True)

        # Test CPDs
        self.assertEqual(len(model_from_str.cpds), 3)

        # Check if all std dev are set
        self.assertIsNotNone(model_from_str.get_cpds("cancer").std)
        self.assertIsNotNone(model_from_str.get_cpds("carry matches").std)
        self.assertIsNotNone(model_from_str.get_cpds("smoking").std)

        # Check variable names
        self.assertEqual(model_from_str.get_cpds("cancer").variable, "cancer")
        self.assertEqual(
            model_from_str.get_cpds("carry matches").variable, "carry matches"
        )
        self.assertEqual(model_from_str.get_cpds("smoking").variable, "smoking")

        # Check evidences
        self.assertEqual(
            sorted(model_from_str.get_cpds("cancer").evidence),
            ["carry matches", "smoking"],
        )
        self.assertEqual(
            sorted(model_from_str.get_cpds("carry matches").evidence), ["smoking"]
        )
        self.assertEqual(sorted(model_from_str.get_cpds("smoking").evidence), [])

        # Check if the betas specified were correctly set
        self.assertEqual(model_from_str.get_cpds("cancer").beta[1], 0.5)
        self.assertEqual(model_from_str.get_cpds("carry matches").beta[1], 0.2)

        # Check if intercepts are 0
        self.assertEqual(model_from_str.get_cpds("cancer").beta[0], 0.0)
        self.assertEqual(model_from_str.get_cpds("carry matches").beta[0], 0.0)
        self.assertEqual(model_from_str.get_cpds("smoking").beta[0], 0.0)

        # Check if std devs are 1
        self.assertEqual(model_from_str.get_cpds("cancer").std, 1)
        self.assertEqual(model_from_str.get_cpds("carry matches").std, 1)
        self.assertEqual(model_from_str.get_cpds("smoking").std, 1)
