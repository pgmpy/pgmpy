import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
import pyro
import pyro.distributions as dist
import torch

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.hybrid.FunctionalCPD import FunctionalCPD
from pgmpy.models import FunctionalBayesianNetwork, LinearGaussianBayesianNetwork


class TestFBNMethods(unittest.TestCase):
    def setUp(self):
        self.model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        self.cpd1 = FunctionalCPD(
            "x1",
            lambda _: dist.Normal(0, 1),
        )
        self.cpd2 = FunctionalCPD(
            "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
        )
        self.cpd3 = FunctionalCPD(
            "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
        )

        self.model.add_cpds(self.cpd1, self.cpd2, self.cpd3)

    def test_cpds_simple(self):
        self.assertEqual("x1", self.cpd1.variable)
        cpd = self.model.get_cpds("x1")
        self.assertEqual(cpd.variable, self.cpd1.variable)
        self.assertEqual(cpd.parents, self.cpd1.parents)
        self.assertEqual(cpd.parents, [])

    def test_add_cpds(self):
        cpd = self.model.get_cpds("x1")
        self.assertEqual(cpd.variable, self.cpd1.variable)

        cpd = self.model.get_cpds("x2")
        self.assertEqual(cpd.variable, self.cpd2.variable)
        self.assertEqual(cpd.parents, self.cpd2.parents)

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

        # Test that duplicate CPDs get replaced.
        self.assertEqual(len(self.model.cpds), 3)
        self.model.add_cpds(self.cpd1)
        self.assertEqual(len(self.model.cpds), 3)

    def test_check_model(self):
        self.assertEqual(self.model.check_model(), True)

        self.model.add_edge("x1", "x4")
        cpd4 = FunctionalCPD(
            "x4", lambda parent: dist.Normal(parent["x2"] * -1 + 4, 3), ["x2"]
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
        fn_cpd1 = FunctionalCPD("x1", lambda _: dist.Normal(1, 1))
        fn_cpd2 = FunctionalCPD(
            "x2",
            lambda parent: dist.Normal(-5 + parent["x1"] * 0.5, 1),
            parents=["x1"],
        )
        fn_cpd3 = FunctionalCPD(
            "x3",
            lambda parent: dist.Normal(4 + parent["x2"] * -1, 1),
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

        cpd1 = FunctionalCPD("exponential", lambda _: dist.Exponential(0.5))

        cpd2 = FunctionalCPD(
            "uniform",
            lambda parent: dist.Uniform(
                parent["exponential"], parent["exponential"] + 2
            ),
            parents=["exponential"],
        )

        cpd3 = FunctionalCPD(
            "lognormal",
            lambda parent: dist.LogNormal(np.log(parent["uniform"]), 1),
            parents=["uniform"],
        )

        cpd4 = FunctionalCPD(
            "gamma",
            lambda parent: dist.Gamma(2.0, parent["lognormal"] / 5),
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

    def test_svi_fit_normal(self):
        alpha = 0.23
        x1 = np.random.normal(1, 2, size=5000)
        x2 = np.random.normal((x1 * alpha) + 5, 1)
        data = pd.DataFrame({"x1": x1, "x2": x2})

        def x1_prior():
            mu = pyro.param("x1_mu", torch.tensor(0.5))
            sigma = pyro.param("x1_sigma", torch.tensor(3.0), constraint=torch.distributions.constraints.positive)
            return dist.Normal(mu, sigma)

        def x2_prior(parent):
            mu = pyro.param("x2_mu", torch.tensor(3.0)) 
            sigma = pyro.param("x2_sigma", torch.tensor(2.0), constraint=torch.distributions.constraints.positive)
            alpha = pyro.param("x2_alpha", torch.tensor(1.0))
            return dist.Normal(mu + (parent["x1"] * alpha), sigma)

        cpd1 = FunctionalCPD('x1', fn=lambda _: x1_prior())
        cpd2 = FunctionalCPD('x2', fn=lambda parent: x2_prior(parent), parents=['x1'])

        self.model.add_cpds(cpd1, cpd2)

        params = self.model.fit(data, method="SVI", learning_rate=0.05, num_steps=2000)

        self.assertIn("x1_mu", params)
        self.assertIn("x1_sigma", params)
        self.assertIn("x2_mu", params)
        self.assertIn("x2_sigma", params)
        self.assertIn("x2_alpha", params)

        self.assertAlmostEqual(params["x1_mu"].mean(), 1, delta=0.2)
        self.assertAlmostEqual(params["x1_sigma"].mean(), 2, delta=0.2)
        self.assertAlmostEqual(params["x2_mu"].mean(), 5, delta=0.2)
        self.assertAlmostEqual(params["x2_sigma"].mean(), 1, delta=0.2)
        self.assertAlmostEqual(params["x2_alpha"].mean(), 0.23, delta=0.2)

    def test_svi_fit_different_distributions(self):
        x1 = np.random.beta(1, 5, size=5000)
        x2 = np.random.poisson(x1 + 5)
        data = pd.DataFrame({"x1": x1, "x2": x2})

        def x1_prior():
            concen1 = pyro.param("x1_concen1", torch.tensor(2.0))
            concen0 = pyro.param("x1_concen0", torch.tensor(3.0))
            return dist.Beta(concen1, concen0)

        def x2_prior(parent):
            rate = pyro.param("x2_rate", dist.Gamma(2, 1))
            return dist.Poisson(rate + parent)

        cpd1 = FunctionalCPD("x1", lambda _: x1_prior())
        cpd2 = FunctionalCPD(
            "x2", lambda parent: x2_prior(parent["x1"]), parents=["x1"]
        )

        self.model.add_cpds(cpd1, cpd2)

        params = self.model.fit(data, method="SVI", learning_rate=0.05, num_steps=2000)

        self.assertIn("x1_concen1", params)
        self.assertIn("x1_concen0", params)
        self.assertIn("x2_rate", params)

        self.assertAlmostEqual(params["x1_concen1"].mean(), 1, delta=0.2)
        self.assertAlmostEqual(params["x1_concen0"].mean(), 5, delta=0.2)
        self.assertAlmostEqual(params["x2_rate"].mean(), 5, delta=0.2)


    def test_mcmc_fit_normal(self):
        alpha = 0.23
        x1 = np.random.normal(1, 2, size=10000)
        x2 = np.random.normal((x1 * alpha) + 5, 1)
        data = pd.DataFrame({"x1": x1, "x2": x2})

        def x1_prior():
            mu = pyro.sample("x1_mu", dist.Normal(0, 10))
            sigma = pyro.sample("x1_sigma", dist.HalfNormal(5))
            return dist.Normal(mu, sigma)

        def x2_prior(parent):
            mu = pyro.sample("x2_mu", dist.Normal(5, 1))
            sigma = pyro.sample("x2_sigma", dist.HalfNormal(2))
            alpha = pyro.sample("x2_alpha", dist.Normal(1, 3))

            return dist.Normal(mu + (alpha * parent['x1']), sigma)

        cpd1 = FunctionalCPD("x1", lambda _: x1_prior())
        cpd2 = FunctionalCPD(
            "x2", lambda parent: x2_prior(parent), parents=["x1"]
        )

        self.model.add_cpds(cpd1, cpd2)
        params = self.model.fit(data, method="MCMC", num_steps=2000)

        self.assertIn("x1_mu", params)
        self.assertIn("x1_sigma", params)
        self.assertIn("x2_mu", params)
        self.assertIn("x2_sigma", params)
        self.assertIn("x2_alpha", params)

        self.assertAlmostEqual(params["x1_mu"].mean(), 1, delta=0.2)
        self.assertAlmostEqual(params["x1_sigma"].mean(), 2, delta=0.2)
        self.assertAlmostEqual(params["x2_mu"].mean(), 5, delta=0.2)
        self.assertAlmostEqual(params["x2_sigma"].mean(), 1, delta=0.2)
        self.assertAlmostEqual(params["x2_alpha"].mean(), 0.23, delta=0.2)

    def test_mcmc_fit_different_distributions(self):
        x1 = np.random.beta(1, 5, size=5000)
        x2 = np.random.poisson(x1 + 5)
        data = pd.DataFrame({"x1": x1, "x2": x2})

        def x1_prior():
            concen1 = pyro.sample("x1_concen1", dist.HalfNormal(3))
            concen0 = pyro.sample("x1_concen0", dist.HalfNormal(4))
            return dist.Beta(concen1, concen0)

        def x2_prior(parent):
            rate = pyro.sample("x2_rate", dist.Gamma(2, 1))
            return dist.Poisson(rate + parent["x1"])

        cpd1 = FunctionalCPD("x1", lambda _: x1_prior())
        cpd2 = FunctionalCPD(
            "x2", lambda parent: x2_prior(parent), parents=["x1"]
        )

        self.model.add_cpds(cpd1, cpd2)

        params = self.model.fit(data, method="MCMC", num_steps=2000)

        self.assertIn("x1_concen1", params)
        self.assertIn("x1_concen0", params)
        self.assertIn("x2_rate", params)

        self.assertAlmostEqual(params["x1_concen1"].mean(), 1, delta=0.2)
        self.assertAlmostEqual(params["x1_concen0"].mean(), 5, delta=0.2)
        self.assertAlmostEqual(params["x2_rate"].mean(), 5, delta=0.2)
