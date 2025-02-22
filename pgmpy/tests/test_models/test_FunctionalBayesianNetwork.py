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

from pgmpy.utils import get_example_model
from pgmpy import config


class TestFBNMethods(unittest.TestCase):
    def setUp(self):
        config.set_backend("torch")
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
        beta = 0.76
        x1 = np.random.normal(1, 2, size=1000)
        x2 = np.random.normal((x1 * alpha) + 5, 1)
        x3 = np.random.normal((x2 * beta) + 3, 1)
        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        def x1_prior():
            mu = pyro.param("x1_mu", torch.tensor(0.5))
            sigma = pyro.param(
                "x1_sigma",
                torch.tensor(3.0),
                constraint=torch.distributions.constraints.positive,
            )
            return dist.Normal(mu, sigma)

        def x2_prior(parent):
            mu = pyro.param("x2_mu", torch.tensor(3.0))
            sigma = pyro.param(
                "x2_sigma",
                torch.tensor(2.0),
                constraint=torch.distributions.constraints.positive,
            )
            alpha = pyro.param("x2_alpha", torch.tensor(1.0))
            return dist.Normal(mu + (parent["x1"] * alpha), sigma)

        def x3_prior(parent):
            mu = pyro.param("x3_mu", torch.tensor(3.0))
            sigma = pyro.param(
                "x3_sigma",
                torch.tensor(2.0),
                constraint=torch.distributions.constraints.positive,
            )
            alpha = pyro.param("x3_beta", torch.tensor(1.0))
            return dist.Normal(mu + (parent["x2"] * alpha), sigma)

        cpd1 = FunctionalCPD("x1", fn=lambda _: x1_prior())
        cpd2 = FunctionalCPD("x2", fn=lambda parent: x2_prior(parent), parents=["x1"])
        cpd3 = FunctionalCPD("x3", fn=lambda parent: x3_prior(parent), parents=["x2"])

        self.model.add_cpds(cpd1, cpd2, cpd3)

        params = self.model.fit(data, method="SVI", learning_rate=0.08, num_steps=100)

        self.assertIn("x1_mu", params)
        self.assertIn("x1_sigma", params)
        self.assertIn("x2_mu", params)
        self.assertIn("x2_sigma", params)
        self.assertIn("x2_alpha", params)
        self.assertIn("x3_mu", params)
        self.assertIn("x3_sigma", params)
        self.assertIn("x3_beta", params)

        self.assertAlmostEqual(params["x1_mu"], 1, delta=0.1)
        self.assertAlmostEqual(params["x1_sigma"], 2, delta=0.1)
        self.assertAlmostEqual(params["x2_mu"], 5, delta=0.1)
        self.assertAlmostEqual(params["x2_sigma"], 1, delta=0.1)
        self.assertAlmostEqual(params["x2_alpha"], 0.23, delta=0.1)
        self.assertAlmostEqual(params["x3_mu"], 3, delta=0.1)
        self.assertAlmostEqual(params["x3_sigma"], 1, delta=0.1)
        self.assertAlmostEqual(params["x3_beta"], 0.76, delta=0.1)

    def test_svi_fit_different_distributions(self):
        x1 = np.random.beta(1, 5, size=1000)
        x2 = np.random.poisson(x1 + 5)
        x3 = np.random.poisson(x2 + 2)
        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        def x1_prior():
            concen1 = pyro.param("x1_concen1", torch.tensor(2.0))
            concen0 = pyro.param("x1_concen0", torch.tensor(3.0))
            return dist.Beta(concen1, concen0)

        def x2_prior(parent):
            rate = pyro.param("x2_rate", dist.Gamma(2, 1))
            return dist.Poisson(rate + parent)

        def x3_prior(parent):
            rate = pyro.param("x3_rate", dist.Gamma(2, 1))
            return dist.Poisson(rate + parent)

        cpd1 = FunctionalCPD("x1", lambda _: x1_prior())
        cpd2 = FunctionalCPD(
            "x2", lambda parent: x2_prior(parent["x1"]), parents=["x1"]
        )
        cpd3 = FunctionalCPD(
            "x3", lambda parent: x3_prior(parent["x2"]), parents=["x2"]
        )

        self.model.add_cpds(cpd1, cpd2, cpd3)

        params = self.model.fit(data, method="SVI", learning_rate=0.08, num_steps=100)

        self.assertIn("x1_concen1", params)
        self.assertIn("x1_concen0", params)
        self.assertIn("x2_rate", params)
        self.assertIn("x3_rate", params)

        self.assertAlmostEqual(params["x1_concen1"], 1, delta=0.1)
        self.assertAlmostEqual(params["x1_concen0"], 5, delta=0.1)
        self.assertAlmostEqual(params["x2_rate"], 5, delta=0.1)
        self.assertAlmostEqual(params["x3_rate"], 3, delta=0.1)

    def test_mcmc_fit_normal(self):
        alpha = 0.23
        beta = 0.81
        x1 = np.random.normal(1, 2, size=700)
        x2 = np.random.normal((x1 * alpha) + 5, 1)
        x3 = np.random.normal((x2 * beta) + 3, 1)
        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        def x1_prior():
            mu = pyro.sample("x1_mu", dist.Normal(0, 10))
            sigma = pyro.sample("x1_sigma", dist.HalfNormal(5))
            return dist.Normal(mu, sigma)

        def x2_prior(parent):
            mu = pyro.sample("x2_mu", dist.Normal(5, 1))
            sigma = pyro.sample("x2_sigma", dist.HalfNormal(2))
            alpha = pyro.sample("x2_alpha", dist.Normal(1, 3))

            return dist.Normal(mu + (alpha * parent["x1"]), sigma)

        def x3_prior(parent):
            mu = pyro.sample("x3_mu", dist.Normal(4, 1))
            sigma = pyro.sample("x3_sigma", dist.HalfNormal(2))
            beta = pyro.sample("x3_beta", dist.Normal(1, 3))

            return dist.Normal(mu + (beta * parent["x2"]), sigma)

        cpd1 = FunctionalCPD("x1", lambda _: x1_prior())
        cpd2 = FunctionalCPD("x2", lambda parent: x2_prior(parent), parents=["x1"])
        cpd3 = FunctionalCPD("x3", lambda parent: x3_prior(parent), parents=["x2"])

        self.model.add_cpds(cpd1, cpd2, cpd3)
        params = self.model.fit(data, method="MCMC", num_steps=100)

        self.assertIn("x1_mu", params)
        self.assertIn("x1_sigma", params)
        self.assertIn("x2_mu", params)
        self.assertIn("x2_sigma", params)
        self.assertIn("x2_alpha", params)
        self.assertIn("x3_mu", params)
        self.assertIn("x3_sigma", params)
        self.assertIn("x3_beta", params)

        self.assertAlmostEqual(params["x1_mu"].mean(), 1, delta=0.1)
        self.assertAlmostEqual(params["x1_sigma"].mean(), 2, delta=0.1)
        self.assertAlmostEqual(params["x2_mu"].mean(), 5, delta=0.1)
        self.assertAlmostEqual(params["x2_sigma"].mean(), 1, delta=0.1)
        self.assertAlmostEqual(params["x2_alpha"].mean(), 0.23, delta=0.1)
        self.assertAlmostEqual(params["x3_mu"].mean(), 3, delta=0.1)
        self.assertAlmostEqual(params["x3_sigma"].mean(), 1, delta=0.1)
        self.assertAlmostEqual(params["x3_beta"].mean(), 0.81, delta=0.1)

    def test_mcmc_fit_different_distributions(self):
        x1 = np.random.beta(1, 5, size=700)
        x2 = np.random.poisson(x1 + 5)
        x3 = np.random.poisson(x2 + 3)
        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        def x1_prior():
            concen1 = pyro.sample("x1_concen1", dist.HalfNormal(3))
            concen0 = pyro.sample("x1_concen0", dist.HalfNormal(4))
            return dist.Beta(concen1, concen0)

        def x2_prior(parent):
            rate = pyro.sample("x2_rate", dist.Gamma(2, 1))
            return dist.Poisson(rate + parent["x1"])

        def x3_prior(parent):
            rate = pyro.sample("x3_rate", dist.Gamma(2, 1))
            return dist.Poisson(rate + parent["x2"])

        cpd1 = FunctionalCPD("x1", lambda _: x1_prior())
        cpd2 = FunctionalCPD("x2", lambda parent: x2_prior(parent), parents=["x1"])
        cpd3 = FunctionalCPD("x3", lambda parent: x3_prior(parent), parents=["x2"])

        self.model.add_cpds(cpd1, cpd2, cpd3)

        params = self.model.fit(data, method="MCMC", num_steps=100)

        self.assertIn("x1_concen1", params)
        self.assertIn("x1_concen0", params)
        self.assertIn("x2_rate", params)
        self.assertIn("x3_rate", params)

        self.assertAlmostEqual(params["x1_concen1"].mean(), 1, delta=0.1)
        self.assertAlmostEqual(params["x1_concen0"].mean(), 5, delta=0.1)
        self.assertAlmostEqual(params["x2_rate"].mean(), 5, delta=0.1)
        self.assertAlmostEqual(params["x3_rate"].mean(), 3, delta=0.1)

    def test_fit_complex_svi(self):
        sim_model = get_example_model("ecoli70")

        b1191 = np.random.normal(1, 2, size=700)
        eutG = np.random.normal(2, 1, size=700)
        fixC = np.random.normal(2 + 3 * b1191, 1)
        ygbD = np.random.normal(3 + 4 * fixC, 1)
        yjbO = np.random.normal(4 + 5 * fixC, 1)
        yceP = np.random.normal(5 + 6 * eutG + 6 * fixC, 1)
        ibpB = np.random.normal(6 + 7 * eutG + 7 * yceP, 1)

        data = pd.DataFrame(
            {
                "b1191": b1191,
                "fixC": fixC,
                "eutG": eutG,
                "ygbD": ygbD,
                "yjbO": yjbO,
                "yceP": yceP,
                "ibpB": ibpB,
            }
        )

        model = FunctionalBayesianNetwork(
            [
                ("b1191", "fixC"),
                ("fixC", "ygbD"),
                ("fixC", "yjbO"),
                ("fixC", "yceP"),
                ("yceP", "ibpB"),
                ("eutG", "yceP"),
                ("eutG", "ibpB"),
            ]
        )

        def positive_param(name, init_value):
            return torch.exp(pyro.param(name, torch.tensor(init_value)))

        def fn_b1191_param():
            mu = pyro.param("b1191_mu", torch.tensor(1.0))
            sigma = positive_param("b1191_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_eutG_param():
            mu = pyro.param("eutG_mu", torch.tensor(1.0))
            sigma = positive_param("eutG_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_fixC_param(parents):
            mu = (
                pyro.param("fixC_inter", torch.tensor(1.0))
                + pyro.param("fixC_alpha", torch.tensor(1.0)) * parents["b1191"]
            )
            sigma = positive_param("fixC_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_ygbD_param(parents):
            mu = (
                pyro.param("ygbD_inter", torch.tensor(1.0))
                + pyro.param("ygbD_alpha", torch.tensor(1.0)) * parents["fixC"]
            )
            sigma = positive_param("ygbD_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_yjbO_param(parents):
            mu = (
                pyro.param("ygbO_inter", torch.tensor(1.0))
                + pyro.param("ygbO_alpha", torch.tensor(1.0)) * parents["fixC"]
            )
            sigma = positive_param("ygbO_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_yceP_param(parents):
            mu = (
                pyro.param("yceP_inter", torch.tensor(1.0))
                + pyro.param("yceP_alpha0", torch.tensor(1.0)) * parents["eutG"]
                + pyro.param("yceP_alpha1", torch.tensor(1.0)) * parents["fixC"]
            )
            sigma = positive_param("yceP_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_ibpB_param(parents):
            mu = (
                pyro.param("ibpB_inter", torch.tensor(1.0))
                + pyro.param("ibpB_alpha0", torch.tensor(1.0)) * parents["eutG"]
                + pyro.param("ibpB_alpha1", torch.tensor(1.0)) * parents["yceP"]
            )
            sigma = positive_param("ibpB_sigma", 1.0)
            return dist.Normal(mu, sigma)

        b1191_cpd = FunctionalCPD("b1191", lambda _: fn_b1191_param())
        eutG_cpd = FunctionalCPD("eutG", lambda _: fn_eutG_param())
        fixC_cpd = FunctionalCPD(
            "fixC", lambda parent: fn_fixC_param(parent), parents=["b1191"]
        )
        ygbD_cpd = FunctionalCPD(
            "ygbD", lambda parent: fn_ygbD_param(parent), parents=["fixC"]
        )
        yjbO_cpd = FunctionalCPD(
            "yjbO", lambda parent: fn_yjbO_param(parent), parents=["fixC"]
        )
        yceP_cpd = FunctionalCPD(
            "yceP", lambda parent: fn_yceP_param(parent), parents=["eutG", "fixC"]
        )
        ibpB_cpd = FunctionalCPD(
            "ibpB", lambda parent: fn_ibpB_param(parent), parents=["eutG", "yceP"]
        )

        model.add_cpds(
            b1191_cpd, eutG_cpd, fixC_cpd, ygbD_cpd, yjbO_cpd, yceP_cpd, ibpB_cpd
        )

        params = model.fit(data, method="SVI", learning_rate=0.08, num_steps=100)

        self.assertIn("b1191_mu", params)
        self.assertIn("eutG_mu", params)
        self.assertIn("fixC_inter", params)
        self.assertIn("fixC_alpha", params)
        self.assertIn("ygbD_inter", params)
        self.assertIn("ygbD_alpha", params)
        self.assertIn("ygbO_inter", params)
        self.assertIn("ygbO_alpha", params)
        self.assertIn("yceP_inter", params)
        self.assertIn("yceP_alpha0", params)
        self.assertIn("yceP_alpha1", params)
        self.assertIn("ibpB_inter", params)
        self.assertIn("ibpB_alpha0", params)
        self.assertIn("ibpB_alpha1", params)

        self.assertAlmostEqual(params["b1191_mu"], 1, delta=0.1)
        self.assertAlmostEqual(params["eutG_mu"], 2, delta=0.1)
        self.assertAlmostEqual(params["fixC_inter"], 2, delta=0.1)
        self.assertAlmostEqual(params["fixC_alpha"], 3, delta=0.1)
        self.assertAlmostEqual(params["ygbD_inter"], 3, delta=0.1)
        self.assertAlmostEqual(params["ygbD_alpha"], 4, delta=0.1)
        self.assertAlmostEqual(params["ygbO_inter"], 4, delta=0.1)
        self.assertAlmostEqual(params["ygbO_alpha"], 5, delta=0.1)
        self.assertAlmostEqual(params["yceP_inter"], 5, delta=0.1)
        self.assertAlmostEqual(params["yceP_alpha0"], 6, delta=0.1)
        self.assertAlmostEqual(params["yceP_alpha1"], 6, delta=0.1)
        self.assertAlmostEqual(params["ibpB_inter"], 7, delta=0.1)
        self.assertAlmostEqual(params["ibpB_alpha0"], 7, delta=0.1)
        self.assertAlmostEqual(params["ibpB_alpha1"], 8, delta=0.1)

    def test_fit_complex_mcmc(self):
        sim_model = get_example_model("ecoli70")

        b1191 = np.random.normal(1, 2, size=700)
        eutG = np.random.normal(2, 1, size=700)
        fixC = np.random.normal(2 + 3 * b1191, 1)
        ygbD = np.random.normal(3 + 4 * fixC, 1)
        yjbO = np.random.normal(4 + 5 * fixC, 1)
        yceP = np.random.normal(5 + 6 * eutG + 6 * fixC, 1)
        ibpB = np.random.normal(6 + 7 * eutG + 7 * yceP, 1)

        data = pd.DataFrame(
            {
                "b1191": b1191,
                "fixC": fixC,
                "eutG": eutG,
                "ygbD": ygbD,
                "yjbO": yjbO,
                "yceP": yceP,
                "ibpB": ibpB,
            }
        )

        model = FunctionalBayesianNetwork(
            [
                ("b1191", "fixC"),
                ("fixC", "ygbD"),
                ("fixC", "yjbO"),
                ("fixC", "yceP"),
                ("yceP", "ibpB"),
                ("eutG", "yceP"),
                ("eutG", "ibpB"),
            ]
        )

        def positive_param(name, init_value):
            return torch.exp(pyro.param(name, torch.tensor(init_value)))

        def fn_b1191_param():
            mu = pyro.sample("b1191_mu", dist.Normal(2, 2))
            sigma = positive_param("b1191_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_eutG_param():
            mu = pyro.sample("eutG_mu", dist.Normal(1.5, 2))
            sigma = positive_param("eutG_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_fixC_param(parents):
            mu = (
                pyro.sample("fixC_inter", dist.Normal(3, 2))
                + pyro.sample("fixC_alpha", dist.Normal(3, 2)) * parents["b1191"]
            )
            sigma = positive_param("fixC_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_ygbD_param(parents):
            mu = (
                pyro.sample("ygbD_inter", dist.Normal(2, 3))
                + pyro.sample("ygbD_alpha", dist.Normal(4, 2)) * parents["fixC"]
            )
            sigma = positive_param("ygbD_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_yjbO_param(parents):
            mu = (
                pyro.sample("ygbO_inter", dist.Normal(3, 2))
                + pyro.sample("ygbO_alpha", dist.Normal(5, 2)) * parents["fixC"]
            )
            sigma = positive_param("ygbO_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_yceP_param(parents):
            mu = (
                pyro.sample("yceP_inter", dist.Normal(6, 2))
                + pyro.sample("yceP_alpha0", dist.Normal(4, 2)) * parents["eutG"]
                + pyro.sample("yceP_alpha1", dist.Normal(6, 3)) * parents["fixC"]
            )
            sigma = positive_param("yceP_sigma", 1.0)
            return dist.Normal(mu, sigma)

        def fn_ibpB_param(parents):
            mu = (
                pyro.sample("ibpB_inter", dist.Normal(6, 3))
                + pyro.sample("ibpB_alpha0", dist.Normal(8, 2)) * parents["eutG"]
                + pyro.sample("ibpB_alpha1", dist.Normal(9, 2)) * parents["yceP"]
            )
            sigma = positive_param("ibpB_sigma", 1.0)
            return dist.Normal(mu, sigma)

        b1191_cpd = FunctionalCPD("b1191", lambda _: fn_b1191_param())
        eutG_cpd = FunctionalCPD("eutG", lambda _: fn_eutG_param())
        fixC_cpd = FunctionalCPD(
            "fixC", lambda parent: fn_fixC_param(parent), parents=["b1191"]
        )
        ygbD_cpd = FunctionalCPD(
            "ygbD", lambda parent: fn_ygbD_param(parent), parents=["fixC"]
        )
        yjbO_cpd = FunctionalCPD(
            "yjbO", lambda parent: fn_yjbO_param(parent), parents=["fixC"]
        )
        yceP_cpd = FunctionalCPD(
            "yceP", lambda parent: fn_yceP_param(parent), parents=["eutG", "fixC"]
        )
        ibpB_cpd = FunctionalCPD(
            "ibpB", lambda parent: fn_ibpB_param(parent), parents=["eutG", "yceP"]
        )

        model.add_cpds(
            b1191_cpd, eutG_cpd, fixC_cpd, ygbD_cpd, yjbO_cpd, yceP_cpd, ibpB_cpd
        )

        params = model.fit(data, method="MCMC", num_steps=100)

        self.assertIn("b1191_mu", params)
        self.assertIn("eutG_mu", params)
        self.assertIn("fixC_inter", params)
        self.assertIn("fixC_alpha", params)
        self.assertIn("ygbD_inter", params)
        self.assertIn("ygbD_alpha", params)
        self.assertIn("ygbO_inter", params)
        self.assertIn("ygbO_alpha", params)
        self.assertIn("yceP_inter", params)
        self.assertIn("yceP_alpha0", params)
        self.assertIn("yceP_alpha1", params)
        self.assertIn("ibpB_inter", params)
        self.assertIn("ibpB_alpha0", params)
        self.assertIn("ibpB_alpha1", params)

        self.assertAlmostEqual(params["b1191_mu"].mean(), 1, delta=0.1)
        self.assertAlmostEqual(params["eutG_mu"].mean(), 2, delta=0.1)
        self.assertAlmostEqual(params["fixC_inter"].mean(), 2, delta=0.1)
        self.assertAlmostEqual(params["fixC_alpha"].mean(), 3, delta=0.1)
        self.assertAlmostEqual(params["ygbD_inter"].mean(), 3, delta=0.1)
        self.assertAlmostEqual(params["ygbD_alpha"].mean(), 4, delta=0.1)
        self.assertAlmostEqual(params["ygbO_inter"].mean(), 4, delta=0.1)
        self.assertAlmostEqual(params["ygbO_alpha"].mean(), 5, delta=0.1)
        self.assertAlmostEqual(params["yceP_inter"].mean(), 5, delta=0.1)
        self.assertAlmostEqual(params["yceP_alpha0"].mean(), 6, delta=0.1)
        self.assertAlmostEqual(params["yceP_alpha1"].mean(), 6, delta=0.1)
        self.assertAlmostEqual(params["ibpB_inter"].mean(), 7, delta=0.1)
        self.assertAlmostEqual(params["ibpB_alpha0"].mean(), 7, delta=0.1)
        self.assertAlmostEqual(params["ibpB_alpha1"].mean(), 8, delta=0.1)


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
