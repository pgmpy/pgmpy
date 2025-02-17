import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.hybrid.FunctionalCPD import FunctionalCPD
from pgmpy.models import FunctionalBayesianNetwork, LinearGaussianBayesianNetwork


class TestFBNMethods(unittest.TestCase):
    def setUp(self):
        self.model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        self.cpd1 = FunctionalCPD("x1", lambda _: np.random.normal(0, 1))
        self.cpd2 = FunctionalCPD(
            "x2", lambda parent: np.random.normal(parent["x1"] + 2.0, 1), parents=["x1"]
        )
        self.cpd3 = FunctionalCPD(
            "x3", lambda parent: np.random.normal(parent["x2"] + 0.3, 2), parents=["x2"]
        )

    def test_cpds_simple(self):
        self.assertEqual("x1", self.cpd1.variable)
        self.model.add_cpds(self.cpd1)
        cpd = self.model.get_cpds("x1")
        self.assertEqual(cpd.variable, self.cpd1.variable)
        self.assertEqual(cpd.parents, self.cpd1.parents)
        self.assertEqual(cpd.parents, [])

    def test_add_cpds(self):
        self.model.add_cpds(self.cpd1)
        cpd = self.model.get_cpds("x1")
        self.assertEqual(cpd.variable, self.cpd1.variable)

        self.model.add_cpds(self.cpd2)
        cpd = self.model.get_cpds("x2")
        self.assertEqual(cpd.variable, self.cpd2.variable)
        self.assertEqual(cpd.parents, self.cpd2.parents)

        self.model.add_cpds(self.cpd3)
        cpd = self.model.get_cpds("x3")
        self.assertEqual(cpd.variable, self.cpd3.variable)
        self.assertEqual(cpd.parents, self.cpd3.parents)

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

    def test_check_model(self):
        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)
        self.assertEqual(self.model.check_model(), True)

        self.model.add_edge("x1", "x4")
        cpd4 = FunctionalCPD(
            "x4", lambda parent: np.random.normal(parent["x2"] * -1 + 4, 3), ["x2"]
        )
        self.model.add_cpds(cpd4)

        self.assertRaises(ValueError, self.model.check_model)

    def test_simulate_linear_gaussian(self):
        lg_model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        lg_cpd1 = LinearGaussianCPD(variable="x1", beta=[1], std=1)
        lg_cpd2 = LinearGaussianCPD(
            variable="x2", beta=[-5, 0.5], std=1, evidence=["x1"]
        )
        lg_cpd3 = LinearGaussianCPD(variable="x3", beta=[4, -1], std=1, evidence=["x2"])
        lg_model.add_cpds(lg_cpd1, lg_cpd2, lg_cpd3)

        fn_model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        fn_cpd1 = FunctionalCPD("x1", lambda _: np.random.normal(1, 1))
        fn_cpd2 = FunctionalCPD(
            "x2",
            lambda parent: np.random.normal(-5 + parent["x1"] * 0.5, 1),
            parents=["x1"],
        )
        fn_cpd3 = FunctionalCPD(
            "x3",
            lambda parent: np.random.normal(4 + parent["x2"] * -1, 1),
            parents=["x2"],
        )
        fn_model.add_cpds(fn_cpd1, fn_cpd2, fn_cpd3)

        n_samples = 5000
        seed = 42
        lg_samples = lg_model.simulate(n=n_samples, seed=seed)
        fn_samples = fn_model.simulate(n_samples=n_samples, seed=seed)

        for var in ["x1", "x2", "x3"]:
            np.testing.assert_allclose(
                lg_samples[var].mean(),
                fn_samples[var].mean(),
                rtol=0.1,
                err_msg=f"Mean mismatch for {var}",
            )
            np.testing.assert_allclose(
                lg_samples[var].std(),
                fn_samples[var].std(),
                rtol=0.1,
                err_msg=f"Standard deviation mismatch for {var}",
            )

    def test_simulate_different_distributions(self):
        model = FunctionalBayesianNetwork(
            [
                ("exponential", "uniform"),
                ("uniform", "lognormal"),
                ("lognormal", "gamma"),
            ]
        )

        cpd1 = FunctionalCPD("exponential", lambda _: np.random.exponential(scale=2.0))

        cpd2 = FunctionalCPD(
            "uniform",
            lambda parent: np.random.uniform(
                low=parent["exponential"], high=parent["exponential"] + 2
            ),
            parents=["exponential"],
        )

        cpd3 = FunctionalCPD(
            "lognormal",
            lambda parent: np.random.lognormal(mean=np.log(parent["uniform"]), sigma=1),
            parents=["uniform"],
        )

        cpd4 = FunctionalCPD(
            "gamma",
            lambda parent: np.random.gamma(shape=2.0, scale=parent["lognormal"] / 5),
            parents=["lognormal"],
        )

        model.add_cpds(cpd1, cpd2, cpd3, cpd4)
        n_samples = 10000
        samples = model.simulate(n_samples=n_samples, seed=42)

        self.assertEqual(len(samples), n_samples)
        self.assertEqual(
            set(samples.columns), {"exponential", "uniform", "lognormal", "gamma"}
        )

        self.assertTrue(np.all(samples["exponential"] >= 0))
        self.assertAlmostEqual(samples["exponential"].mean(), 2.0, delta=0.2)

        self.assertTrue(np.all(samples["uniform"] >= samples["exponential"]))
        self.assertTrue(np.all(samples["uniform"] <= samples["exponential"] + 2))

        self.assertTrue(np.all(samples["lognormal"] > 0))
        self.assertTrue(np.all(samples["gamma"] > 0))


class TestFBNCreation(unittest.TestCase):
    def test_class_init_with_adj_matrix_dict_of_dict(self):
        adj = {"a": {"b": 4, "c": 3}, "b": {"c": 2}}
        self.graph = FunctionalBayesianNetwork(adj, latents=set(["a"]))
        self.assertEqual(self.graph.latents, set("a"))
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])
        self.assertEqual(self.graph.adj["a"]["c"]["weight"], 3)

    def test_class_init_with_adj_matrix_dict_of_list(self):
        adj = {"a": ["b", "c"], "b": ["c"]}
        self.graph = FunctionalBayesianNetwork(adj, latents=set(["a"]))
        self.assertEqual(self.graph.latents, set("a"))
        self.assertListEqual(sorted(self.graph.nodes()), ["a", "b", "c"])

    def test_class_init_with_pd_adj_df(self):
        df = pd.DataFrame([[0, 3], [0, 0]])
        self.graph = FunctionalBayesianNetwork(df, latents=set([0]))
        self.assertEqual(self.graph.latents, set([0]))
        self.assertListEqual(sorted(self.graph.nodes()), [0, 1])
        self.assertEqual(self.graph.adj[0][1]["weight"], {"weight": 3})


class TestDAGParser(unittest.TestCase):
    def test_from_lavaan(self):
        model_str = "ind60 =~ x1"
        model_from_str = FunctionalBayesianNetwork.from_lavaan(string=model_str)
        expected_edges = set([("ind60", "x1")])
        self.assertEqual(set(model_from_str.edges()), expected_edges)

    def test_from_dagitty(self):
        model_str = """dag{X -> Y}"""
        model_from_str = FunctionalBayesianNetwork.from_dagitty(string=model_str)
        expected_edges = set([("X", "Y")])
        self.assertEqual(set(model_from_str.edges()), expected_edges)
